"""Leakage tests for the feature pipeline (src/features.py).

The per-card history features must use only *past* transactions, and fitted
encoders must learn only from the training years. These are the checks from
notebook 02, written as tests:

- Truncation: deleting or changing every future transaction leaves the history
  features of earlier rows unchanged.
- Brute-force recount: the vectorised window features match a slow, obviously
  correct loop over each card's earlier transactions.
- Encoders: the frequency encoder learns only from training rows.

Most tests use a small synthetic dataset, so they run anywhere in a few seconds.
`test_truncation_on_real_data` repeats the truncation check on the cached IBM
data and is skipped when that cache isn't there.

    pytest tests/
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import features as F
from src.data import CACHE_DIR

HISTORY = F.CARD_HISTORY_FEATURES
CUTOFF = pd.Timestamp("2015-01-01")


def history(df: pd.DataFrame) -> pd.DataFrame:
    return F.add_card_history_features(df, F.add_transaction_features(df))[HISTORY]


@pytest.fixture(scope="module")
def synthetic() -> pd.DataFrame:
    """A few cards with bursts of activity, errors, refunds and repeat merchants,
    with every card's timeline centred on CUTOFF so each card has both past and
    future rows. Rows are shuffled across cards, as in the raw file."""
    rng = np.random.default_rng(42)
    rows = []
    for user in range(6):
        for card in range(2):
            n = int(rng.integers(40, 120))
            # Gaps from seconds to weeks, so the 1h and 24h windows hold 0 to many rows
            gaps = rng.choice([30, 600, 3000, 20_000, 90_000, 2_000_000], size=n)
            offsets = np.cumsum(gaps) - gaps.sum() // 2
            t = CUTOFF + pd.Timedelta(seconds=17) + pd.to_timedelta(offsets, unit="s")
            for ts in t.floor("min"):
                rows.append({
                    "User": user, "Card": card, "Transaction Datetime": ts,
                    "Year": ts.year, "Hour": ts.hour,
                    "Amount": float(np.round(rng.normal(60, 80), 2)),
                    "Use Chip": rng.choice(["Chip Transaction", "Swipe Transaction", "Online Transaction"]),
                    "Merchant Name": int(rng.integers(0, 15)),
                    "Merchant City": rng.choice(["Austin", "Paris", "Reno"]),
                    "Merchant State": rng.choice(["TX", "NV", "France", None]),
                    "Zip": rng.choice([78701.0, 89501.0, np.nan]),
                    "MCC": int(rng.choice([5411, 5812, 4829])),
                    "Errors?": rng.choice([None, None, None, "Bad PIN", "Insufficient Balance"]),
                })
    df = pd.DataFrame(rows)
    return df.sample(frac=1, random_state=0).reset_index(drop=True)


@pytest.fixture(scope="module")
def past(synthetic) -> np.ndarray:
    """Rows before CUTOFF: the "past" part of the synthetic data."""
    return (synthetic["Transaction Datetime"] < CUTOFF).to_numpy()


def test_synthetic_data_spans_cutoff(synthetic, past):
    """Guard: the truncation tests are only meaningful if every card has rows on
    both sides of the cutoff."""
    per_card = pd.Series(past).groupby([synthetic["User"], synthetic["Card"]]).agg(["min", "max"])
    assert (~per_card["min"] & per_card["max"]).all()


def test_truncation_drop_future_rows(synthetic, past):
    """History features of past rows are identical with and without the future rows."""
    full = history(synthetic)[past].reset_index(drop=True)
    cut = history(synthetic[past].reset_index(drop=True))
    pd.testing.assert_frame_equal(full, cut)


def test_truncation_perturb_future_rows(synthetic, past):
    """Changing every future amount, error and merchant leaves past rows unchanged."""
    changed = synthetic.copy()
    changed.loc[~past, "Amount"] = changed.loc[~past, "Amount"] * 10 + 999
    changed.loc[~past, "Errors?"] = "Technical Glitch"
    changed.loc[~past, "Merchant Name"] = changed.loc[~past, "Merchant Name"] + 1000
    a = history(synthetic)[past]
    b = history(changed)[past]
    pd.testing.assert_frame_equal(a, b)
    # Guard: the perturbation must actually reach the future rows' features
    assert not history(synthetic)[~past].equals(history(changed)[~past])


def test_window_features_match_brute_force(synthetic):
    """Every row's 1h/24h counts, 24h amount and 24h errors match a slow recount
    over the card's earlier transactions, and so do the other history features."""
    h = history(synthetic)
    df = synthetic.assign(_row=np.arange(len(synthetic)))
    for _, card in df.groupby(["User", "Card"]):
        # Same order the pipeline uses: time, then original file order for ties
        card = card.sort_values(["Transaction Datetime", "_row"])
        times = card["Transaction Datetime"].tolist()
        amounts = card["Amount"].tolist()
        errors = card["Errors?"].notna().tolist()
        merchants = card["Merchant Name"].tolist()
        for i, idx in enumerate(card.index):
            t = times[i]
            in_1h = [j for j in range(i) if times[j] >= t - pd.Timedelta("1h")]
            in_24h = [j for j in range(i) if times[j] >= t - pd.Timedelta("24h")]
            got = h.loc[idx]
            assert got["card_txn_index"] == i
            assert got["card_txns_1h"] == len(in_1h)
            assert got["card_txns_24h"] == len(in_24h)
            assert got["card_errors_24h"] == sum(errors[j] for j in in_24h)
            assert got["card_amount_24h"] == pytest.approx(sum(amounts[j] for j in in_24h), abs=0.05)
            assert got["is_new_merchant_for_card"] == int(merchants[i] not in merchants[:i])
            assert got["prev_txn_had_error"] == (int(errors[i - 1]) if i > 0 else 0)
            if i == 0:
                assert np.isnan(got["secs_since_prev_txn"]) and np.isnan(got["amt_vs_card_mean"])
            else:
                assert got["secs_since_prev_txn"] == (t - times[i - 1]).total_seconds()
                mean = np.mean(amounts[:i])
                if mean != 0:
                    assert got["amt_vs_card_mean"] == pytest.approx(amounts[i] / mean, rel=1e-4)


def test_frequency_encoder_learns_only_from_train(synthetic, past):
    """Encodings come from the training rows alone: IDs seen only later get 0,
    and the learned map is exactly the training frequencies."""
    train = synthetic[past]
    later = synthetic[~past].copy()
    later["Merchant Name"] = later["Merchant Name"] + 1000   # IDs never seen in training

    enc = F.FrequencyEncoder(F.FREQ_COLUMNS).fit(train)
    out = enc.transform(later, F.add_transaction_features(later))
    assert (out["merchant_freq"] == 0).all()

    expected = train["Merchant Name"].value_counts(normalize=True)
    pd.testing.assert_series_equal(enc.maps_["Merchant Name"], expected)


REAL_CACHE = CACHE_DIR / "transactions_clean.parquet"


@pytest.mark.skipif(not REAL_CACHE.exists(), reason="needs data/cache/transactions_clean.parquet (run notebook 01)")
def test_truncation_on_real_data():
    """Notebook 02's truncation test on 50 real users: rows before 2015 have identical
    history features whether or not 2015-2020 is present."""
    df = pd.read_parquet(REAL_CACHE, filters=[("User", "<", 50)]).reset_index(drop=True)
    past = (df["Year"] < 2015).to_numpy()
    full = history(df)[past].reset_index(drop=True)
    cut = history(df[past].reset_index(drop=True))
    pd.testing.assert_frame_equal(full, cut)
