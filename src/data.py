"""Loading, cleaning and caching for the IBM synthetic credit card dataset.

Notebooks import from here so every stage cleans the data the same way:

    from src.data import load_clean
    df = load_clean()             # full 24.4M rows, cached as Parquet after first run
    df = load_clean(sample=True)  # single-user sample, for quick dry runs

Paths resolve from the project root, so this works from any working directory.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
CACHE_DIR = PROJECT_ROOT / "data" / "cache"

TRANSACTIONS_FULL = RAW_DIR / "credit_card_transactions-ibm_v2.csv"
TRANSACTIONS_SAMPLE = RAW_DIR / "User0_credit_card_transactions.csv"
USERS_FILE = RAW_DIR / "sd254_users.csv"
CARDS_FILE = RAW_DIR / "sd254_cards.csv"

TARGET = "Is Fraud?"


def load_raw(sample: bool = False) -> pd.DataFrame:
    """Read the raw transactions CSV untouched."""
    path = TRANSACTIONS_SAMPLE if sample else TRANSACTIONS_FULL
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. See data/README.md for download steps.")
    return pd.read_csv(path)


def clean_transactions(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Fix dtypes and parse time fields. Does not drop or impute anything —
    data quality decisions (drop/impute/flag) belong in later, explicit steps,
    not silently inside a cleaning function."""
    d = df_raw.copy()

    # Amount: strip '$' and convert to numeric (unparseable values become NaN)
    d["Amount"] = d["Amount"].astype(str).str.replace("$", "", regex=False)
    d["Amount"] = pd.to_numeric(d["Amount"], errors="coerce").astype("float32")

    # Time: strip stray whitespace, then extract Hour/Minute with a string split
    # (faster than full datetime parsing at this scale)
    d["Time"] = d["Time"].astype(str).str.strip()
    d["Hour"] = d["Time"].str.split(":").str[0].astype("int8")
    d["Minute"] = d["Time"].str.split(":").str[1].astype("int8")

    # Real datetime column for chronological sorting / time-series work.
    # An explicit format avoids per-row format inference on 24M rows.
    d["Transaction Datetime"] = pd.to_datetime(
        d["Year"].astype(str) + "-" +
        d["Month"].astype(str).str.zfill(2) + "-" +
        d["Day"].astype(str).str.zfill(2) + " " + d["Time"],
        format="%Y-%m-%d %H:%M",
        errors="coerce",
    )

    # Small-range integers: downcast
    for col in ["User", "Card", "Year", "Month", "Day", "MCC"]:
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce")
            if d[col].dropna().between(0, 127).all():
                d[col] = d[col].astype("int8")
            elif d[col].dropna().between(0, 32767).all():
                d[col] = d[col].astype("int16")
            else:
                d[col] = d[col].astype("int32")

    # Merchant Name is an ID, not a quantity — huge range, keep as int64
    if "Merchant Name" in d.columns:
        d["Merchant Name"] = pd.to_numeric(d["Merchant Name"], errors="coerce")

    # Zip is a label, not a number — keep as string to preserve leading zeros
    if "Zip" in d.columns:
        d["Zip"] = d["Zip"].astype("Int64").astype(str).replace("<NA>", np.nan)

    # Low-cardinality repeating strings -> category
    for col in ["Use Chip", "Merchant City", "Merchant State", "Errors?", "Is Fraud?"]:
        if col in d.columns:
            d[col] = d[col].astype("category")

    return d


def load_clean(sample: bool = False, refresh: bool = False) -> pd.DataFrame:
    """Cleaned transactions, cached as Parquet so later loads skip the CSV parse.

    Pass refresh=True after changing clean_transactions() to rebuild the cache.
    """
    cache = CACHE_DIR / ("transactions_clean_sample.parquet" if sample else "transactions_clean.parquet")
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)
    df = clean_transactions(load_raw(sample))
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, index=False)
    return df


def _dollars(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace("$", "", regex=False), errors="coerce")


def load_users() -> pd.DataFrame:
    """User attributes keyed by `User` (the row index of sd254_users.csv).

    Note: these are snapshots taken at data-generation time, not point-in-time values.
    """
    users = pd.read_csv(USERS_FILE).reset_index().rename(columns={"index": "User"})
    for col in ["Per Capita Income - Zipcode", "Yearly Income - Person", "Total Debt"]:
        users[col] = _dollars(users[col])
    users["User"] = users["User"].astype("int16")
    cols = ["User", "Current Age", "Gender", "Yearly Income - Person",
            "Total Debt", "FICO Score", "Num Credit Cards"]
    return users[cols]  # name and address dropped: PII, no modeling value


def load_cards() -> pd.DataFrame:
    """Card attributes keyed by (`User`, `Card`). Card number and CVV are dropped (PII)."""
    cards = pd.read_csv(CARDS_FILE).rename(columns={"CARD INDEX": "Card"})
    cards["Credit Limit"] = _dollars(cards["Credit Limit"])
    cards["Acct Open Date"] = pd.to_datetime(cards["Acct Open Date"], format="%m/%Y")
    cards = cards.astype({"User": "int16", "Card": "int8"})
    cols = ["User", "Card", "Card Brand", "Card Type", "Has Chip",
            "Credit Limit", "Acct Open Date", "Card on Dark Web"]
    return cards[cols]


class MergeError(ValueError):
    """Raised when a join would silently corrupt the transaction table."""


def merge_report(df: pd.DataFrame, users: pd.DataFrame, cards: pd.DataFrame) -> pd.DataFrame:
    """Check the keys *before* joining. Returns one row per check, with a status:

    - FAIL: the join would be wrong (duplicate keys fan out rows; type
      mismatches silently match nothing; unmatched transactions get NaN attributes).
    - INFO: worth knowing but harmless (e.g. cards that never transact).
    """
    rows = []

    def add(check, value, fail_if_nonzero=True):
        status = "FAIL" if (fail_if_nonzero and value) else ("INFO" if not fail_if_nonzero else "OK")
        rows.append({"check": check, "value": value, "status": status})

    # 1. Duplicate keys in the dimension tables would multiply transaction rows
    add("duplicate User keys in users", int(users["User"].duplicated().sum()))
    add("duplicate (User, Card) keys in cards", int(cards.duplicated(["User", "Card"]).sum()))

    # 2. Key dtype families must agree, or pandas raises / matches nothing
    for key in ["User", "Card"]:
        other = users if key == "User" else cards
        mismatch = df[key].dtype.kind != other[key].dtype.kind
        add(f"dtype mismatch on {key} ({df[key].dtype} vs {other[key].dtype})", int(mismatch))

    # 3. Transactions whose keys have no match (left join would fill NaN)
    tx_users = pd.Index(df["User"].unique())
    tx_cards = pd.MultiIndex.from_frame(df[["User", "Card"]].drop_duplicates())
    card_keys = pd.MultiIndex.from_frame(cards[["User", "Card"]])
    missing_users = tx_users.difference(users["User"])
    missing_cards = tx_cards.difference(card_keys)
    add("transaction users missing from users table", len(missing_users))
    add("transaction cards missing from cards table", len(missing_cards))
    if len(missing_cards):
        n_rows = df.set_index(["User", "Card"]).index.isin(missing_cards).sum()
        add("  -> transactions affected", int(n_rows))

    # 4. Informational: dimension rows with no transactions
    add("users with no transactions", len(pd.Index(users["User"]).difference(tx_users)), False)
    add("cards with no transactions", len(card_keys.difference(tx_cards)), False)
    return pd.DataFrame(rows)


def join_users_cards(df: pd.DataFrame, cols: list[str] | None = None, strict: bool = True,
                     verbose: bool = False) -> pd.DataFrame:
    """Left-join user and card attributes onto transactions, with error handling.

    Efficiency: only the key columns (or `cols`) of the 24M-row table go into the
    merge, and the small dimension tables are merged onto it — never the reverse.
    With strict=True any FAIL in `merge_report` raises MergeError before joining.
    After joining, it checks the row count and that no attribute became null
    for a matched row.
    """
    users, cards = load_users(), load_cards()
    report = merge_report(df, users, cards)
    if verbose:
        print(report.to_string(index=False))
    failures = report[report["status"] == "FAIL"]
    if strict and len(failures):
        raise MergeError("Join pre-checks failed:\n" + failures.to_string(index=False))

    base = df[["User", "Card"] + [c for c in (cols or []) if c not in ("User", "Card")]]
    out = (base
           .merge(users, on="User", how="left", validate="many_to_one")
           .merge(cards, on=["User", "Card"], how="left", validate="many_to_one", indicator=True))

    if len(out) != len(df):
        raise MergeError(f"Join changed the row count: {len(df):,} -> {len(out):,}")
    unmatched = int((out["_merge"] != "both").sum())
    if strict and unmatched:
        raise MergeError(f"{unmatched:,} transactions found no card")
    # Nulls the join introduced (beyond nulls already present in the source tables)
    for name, table in [("users", users), ("cards", cards)]:
        for col in table.columns:
            if col in ("User", "Card") or table[col].isna().any():
                continue
            new_nulls = int(out.loc[out["_merge"] == "both", col].isna().sum())
            if new_nulls:
                raise MergeError(f"{new_nulls:,} matched rows have null {name}.{col}")
    out.index = df.index                     # align with the caller's rows
    return out.drop(columns="_merge")
