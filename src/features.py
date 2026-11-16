"""Phase 1 feature engineering and chronological split.

Leakage rules this module follows:
- Per-card history features use only transactions *before* the current one
  (data sorted by card and time; ties at the same minute keep file order).
- No feature uses fraud labels — in production, labels arrive weeks late.
- Anything fitted to data (frequency encoders, category levels) is fitted on
  the training period only, via `FrequencyEncoder.fit(train)`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data import TARGET, join_users_cards

# ---------------------------------------------------------------- split

TRAIN_END_YEAR = 2017   # train: 1991-2017
VAL_YEAR = 2018         # validation: 2018 (2,491 frauds)
TEST_YEAR = 2019        # test: 2019 (2,087 frauds)
# 2020 (Jan-Feb) is excluded: 0 frauds in 336k rows, i.e. labels look missing.

ERROR_TYPES = ["Insufficient Balance", "Bad PIN", "Technical Glitch", "Bad Card Number",
               "Bad CVV", "Bad Expiration", "Bad Zipcode"]

# Feature groups. Ablations in Phase 5 add/remove whole groups.
BASE_FEATURES = [
    "amount_signed_log", "is_refund", "is_zero_amount",
    "hour", "day_of_week",
    "use_chip", "is_online", "is_international", "mcc",
    "has_error", *[f"err_{e.lower().replace(' ', '_')}" for e in ERROR_TYPES],
    "merchant_freq", "merchant_city_freq", "zip_freq",
]
CARD_HISTORY_FEATURES = [
    "card_txn_index", "secs_since_prev_txn", "is_new_merchant_for_card",
    "amt_vs_card_mean", "amt_card_zscore",
    "card_txns_1h", "card_txns_24h", "card_amount_24h", "card_errors_24h",
    "prev_txn_had_error",
]
CARD_ATTRIBUTE_FEATURES = ["card_brand", "card_type", "card_has_chip",
                           "credit_limit", "card_age_days", "amount_to_limit"]
# Held out of the default set: user snapshots may leak (Phase 5 ablation),
# Merchant State is the suspected synthetic shortcut (Phase 5 ablation).
USER_SNAPSHOT_FEATURES = ["user_age", "user_income", "user_debt", "fico_score", "num_credit_cards"]
SHORTCUT_FEATURES = ["merchant_state"]
# is_international is derived from Merchant State (13x fraud lift on train), so it carries
# the same shortcut. It stays in the default set as the plan specifies, but the Phase 5
# "no geography" ablation must drop GEO_DERIVED_FEATURES too, not just merchant_state.
GEO_DERIVED_FEATURES = ["is_international"]

DEFAULT_FEATURES = BASE_FEATURES + CARD_HISTORY_FEATURES + CARD_ATTRIBUTE_FEATURES
CATEGORICAL_FEATURES = ["use_chip", "mcc", "card_brand", "card_type", "merchant_state"]


def split_name(year: pd.Series) -> pd.Series:
    """Map transaction year to 'train' / 'val' / 'test' / 'excluded'."""
    return pd.Series(
        np.select([year <= TRAIN_END_YEAR, year == VAL_YEAR, year == TEST_YEAR],
                  ["train", "val", "test"], default="excluded"),
        index=year.index,
    )


# ---------------------------------------------------------------- per-row features

def add_transaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features computable from the transaction row alone (no history, no fitting)."""
    out = pd.DataFrame(index=df.index)
    amt = df["Amount"].astype("float32")
    out["amount"] = amt
    out["amount_signed_log"] = (np.sign(amt) * np.log1p(np.abs(amt))).astype("float32")
    out["is_refund"] = (amt < 0).astype("int8")
    out["is_zero_amount"] = (amt == 0).astype("int8")

    out["hour"] = df["Hour"].astype("int8")
    out["day_of_week"] = df["Transaction Datetime"].dt.dayofweek.astype("int8")

    out["use_chip"] = df["Use Chip"].astype("category")
    out["is_online"] = (df["Use Chip"] == "Online Transaction").astype("int8")
    state = df["Merchant State"].astype("string")
    # Domestic merchants carry a 2-letter US code; foreign ones carry a country name.
    out["is_international"] = (state.notna() & (state.str.len() > 2)).fillna(False).astype("int8")
    # Shortcut candidate, kept only for the ablation. Online rows have no state:
    # make that explicit instead of leaving NaN.
    out["merchant_state"] = state.fillna("ONLINE_OR_MISSING").astype("category")
    # MCC is a label: string levels (Parquet turns integer-level categories back into ints)
    out["mcc"] = df["MCC"].astype(str).astype("category")

    errors = df["Errors?"].astype("string")
    out["has_error"] = errors.notna().astype("int8")
    for e in ERROR_TYPES:
        col = f"err_{e.lower().replace(' ', '_')}"
        out[col] = errors.str.contains(e, regex=False).fillna(False).astype("int8")
    return out


# ---------------------------------------------------------------- per-card history

def _group_prior_cumsum(v: np.ndarray, key: np.ndarray, new_group: np.ndarray) -> np.ndarray:
    """Sum of `v` over the earlier rows of the same group (0 for a group's first row).

    The cumulative sum restarts for every card, so a card's values depend only on
    that card's own past, not on float rounding from other cards' rows.
    """
    within = pd.Series(v, dtype="float64").groupby(key, sort=False).cumsum().to_numpy()
    prior = np.r_[0.0, within[:-1]]
    prior[new_group] = 0.0
    return prior


def _window_stats(key: np.ndarray, t: np.ndarray, new_group: np.ndarray,
                  window_s: int, values: list[np.ndarray]):
    """For rows sorted by (key, t): count of earlier rows of the same key within
    `window_s` seconds, plus the sum of each array in `values` over those rows.

    Works on the whole array at once: key and time are packed into one
    increasing int64, so a binary search never crosses into another card.
    """
    packed = key.astype(np.int64) * 10**10 + t        # t < 1e10 s, so cards never overlap
    start = np.searchsorted(packed, packed - window_s, side="left")
    idx = np.arange(len(packed))
    counts = idx - start                               # rows in [t - w, t), excluding self
    sums = []
    for v in values:
        prior = _group_prior_cumsum(v, key, new_group)
        sums.append(prior[idx] - prior[start])         # start is always in the same card
    return counts, sums


def add_card_history_features(df: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    """Behavioral features from the card's own *past* transactions only."""
    order = np.lexsort((np.arange(len(df)), df["Transaction Datetime"].values,
                        df["Card"].values, df["User"].values))
    d = df.iloc[order]
    card_key = (d["User"].astype(np.int64) * 100 + d["Card"].astype(np.int64)).to_numpy()
    t = ((d["Transaction Datetime"] - pd.Timestamp("1990-01-01")).dt.total_seconds()).astype(np.int64).to_numpy()
    amt = d["Amount"].astype(np.float64).to_numpy()
    err = d["Errors?"].notna().astype(np.float64).to_numpy()

    new_card = np.r_[True, card_key[1:] != card_key[:-1]]
    group_start = np.maximum.accumulate(np.where(new_card, np.arange(len(d)), 0))
    n_prev = np.arange(len(d)) - group_start            # transactions seen before this one

    h = pd.DataFrame(index=d.index)
    h["card_txn_index"] = n_prev.astype("int32")

    prev_t = np.r_[0, t[:-1]]
    h["secs_since_prev_txn"] = np.where(new_card, np.nan, t - prev_t).astype("float32")

    prev_err = np.r_[0.0, err[:-1]]
    h["prev_txn_had_error"] = np.where(new_card, 0, prev_err).astype("int8")

    h["is_new_merchant_for_card"] = (~d.duplicated(["User", "Card", "Merchant Name"])).astype("int8").to_numpy()

    # Expanding mean / std of *previous* amounts on the card
    s1 = _group_prior_cumsum(amt, card_key, new_card)
    s2 = _group_prior_cumsum(amt**2, card_key, new_card)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = s1 / n_prev
        var = np.maximum(s2 / n_prev - mean**2, 0)
        std = np.sqrt(var)
        h["amt_vs_card_mean"] = np.where(n_prev > 0, amt / np.where(mean == 0, np.nan, mean), np.nan).astype("float32")
        h["amt_card_zscore"] = np.where(n_prev > 1, (amt - mean) / np.where(std == 0, np.nan, std), np.nan).astype("float32")

    c1h, _ = _window_stats(card_key, t, new_card, 3600, [])
    c24h, (amt24, err24) = _window_stats(card_key, t, new_card, 86400, [amt, err])
    h["card_txns_1h"] = c1h.astype("int32")
    h["card_txns_24h"] = c24h.astype("int32")
    h["card_amount_24h"] = amt24.astype("float32")
    h["card_errors_24h"] = err24.astype("int32")

    return feats.join(h)  # join on index restores the original row order


# ---------------------------------------------------------------- joined attributes

def add_card_and_user_attributes(df: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    """Card attributes (fixed per card) and user snapshots (ablation only)."""
    # Checked join: raises MergeError on duplicate/unmatched keys or row-count changes
    cards = users = join_users_cards(df)

    out = feats.copy()
    out["card_brand"] = pd.Categorical(cards["Card Brand"])
    out["card_type"] = pd.Categorical(cards["Card Type"])
    out["card_has_chip"] = (cards["Has Chip"] == "YES").astype("int8").to_numpy()
    out["credit_limit"] = cards["Credit Limit"].astype("float32").to_numpy()
    # Point-in-time: days the account had been open when the transaction happened
    out["card_age_days"] = ((df["Transaction Datetime"].to_numpy() - cards["Acct Open Date"].to_numpy())
                            / np.timedelta64(1, "D")).astype("float32")
    with np.errstate(divide="ignore", invalid="ignore"):
        out["amount_to_limit"] = (out["amount"] / out["credit_limit"].replace(0, np.nan)).astype("float32")

    out["user_age"] = users["Current Age"].astype("float32").to_numpy()
    out["user_income"] = users["Yearly Income - Person"].astype("float32").to_numpy()
    out["user_debt"] = users["Total Debt"].astype("float32").to_numpy()
    out["fico_score"] = users["FICO Score"].astype("float32").to_numpy()
    out["num_credit_cards"] = users["Num Credit Cards"].astype("float32").to_numpy()
    return out


# ---------------------------------------------------------------- fitted encoders

class FrequencyEncoder:
    """Replace a high-cardinality ID with its share of *training* transactions.
    IDs never seen in training get 0."""

    def __init__(self, columns: dict[str, str]):
        self.columns = columns          # {raw column: output feature name}
        self.maps_: dict[str, pd.Series] = {}

    def fit(self, raw_train: pd.DataFrame) -> "FrequencyEncoder":
        for col in self.columns:
            self.maps_[col] = raw_train[col].value_counts(normalize=True, dropna=True)
        return self

    def transform(self, raw: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
        out = feats.copy()
        for col, name in self.columns.items():
            out[name] = raw[col].map(self.maps_[col]).astype("float32").fillna(0).to_numpy()
        return out


FREQ_COLUMNS = {"Merchant Name": "merchant_freq", "Merchant City": "merchant_city_freq", "Zip": "zip_freq"}


def align_categories(train: pd.DataFrame, other: pd.DataFrame) -> pd.DataFrame:
    """Give categorical columns the training-set levels; unseen levels become NaN."""
    out = other.copy()
    for col in CATEGORICAL_FEATURES:
        if col in out:
            levels = train[col].cat.remove_unused_categories().cat.categories
            out[col] = pd.Categorical(out[col].astype(object), categories=levels)
    return out


# ---------------------------------------------------------------- pipeline

def build_features(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Full Phase 1 pipeline: unfitted features on all rows, then split, then
    fit encoders on train and apply to every split. Returns {'train','val','test'}."""
    feats = add_transaction_features(df)
    feats = add_card_history_features(df, feats)   # uses all past rows, including pre-split history
    feats = add_card_and_user_attributes(df, feats)
    feats["year"] = df["Year"].astype("int16")
    feats["is_fraud"] = (df[TARGET] == "Yes").astype("int8")

    split = split_name(df["Year"])
    enc = FrequencyEncoder(FREQ_COLUMNS).fit(df[split == "train"])

    out = {}
    for name in ["train", "val", "test"]:
        m = (split == name).to_numpy()
        out[name] = enc.transform(df[m], feats[m])
    for name in ["val", "test"]:
        out[name] = align_categories(out["train"], out[name])
    return out
