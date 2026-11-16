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


def join_users_cards(df: pd.DataFrame) -> pd.DataFrame:
    """Left-join user and card attributes onto transactions, checking row count is unchanged."""
    out = (df
           .merge(load_users(), on="User", how="left", validate="many_to_one")
           .merge(load_cards(), on=["User", "Card"], how="left", validate="many_to_one"))
    assert len(out) == len(df), "left join changed the row count"
    return out
