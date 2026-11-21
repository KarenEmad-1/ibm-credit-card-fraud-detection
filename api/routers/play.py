"""Interactive screens: Beat-the-model game, card attack timelines and alert triage.

All of them read saved scores only. Labels for the game and triage are kept out of
the first response and only sent by the matching `reveal` call, so the page has to
commit to an answer first.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from api.store import SplitData, store

router = APIRouter(prefix="/api", tags=["play"])

SHOW = ["txn_id", "user", "card", "amount", "merchant", "city", "state", "mcc", "channel", "errors"]


def _model(d: SplitData, key: str | None) -> str:
    if key is None:
        return next((k for k in d.model_keys if k.startswith("CatBoost")), d.model_keys[0])
    if key not in d.models:
        raise HTTPException(404, f"unknown model '{key}'")
    return key


def _records(d: SplitData, idx: np.ndarray, extra: dict | None = None) -> list[dict]:
    df = d.tx.iloc[idx][SHOW].copy()
    df.insert(1, "ts", d.ts_sec[idx])
    for k, v in (extra or {}).items():
        df[k] = v
    return df.to_dict(orient="records")


# ----------------------------------------------------------------------------- per-card index
@lru_cache(maxsize=4)
def _card_index(split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(row positions grouped by card in time order, group start offsets, card keys)."""
    d = store.get(split)
    key = d.tx["user"].to_numpy(np.int64) * 100 + d.tx["card"].to_numpy(np.int64)
    order = np.argsort(key, kind="stable")                # tx is chronological, so groups stay in time order
    k_sorted = key[order]
    starts = np.flatnonzero(np.r_[True, k_sorted[1:] != k_sorted[:-1]])
    return order, np.r_[starts, len(order)], k_sorted[starts]


def _card_rows(split: str, user: int, card: int) -> np.ndarray:
    order, bounds, keys = _card_index(split)
    g = np.searchsorted(keys, user * 100 + card)
    if g >= len(keys) or keys[g] != user * 100 + card:
        raise HTTPException(404, "card not found")
    return order[bounds[g]:bounds[g + 1]]


# ----------------------------------------------------------------------------- beat the model
@router.get("/game/round")
def game_round(seed: int = 0, split: str | None = None, model: str | None = None):
    """One transaction to judge, with the card's previous 6 transactions as context.

    Mix: 45% frauds, 30% ordinary legit, 25% legit transactions the model wrongly flagged,
    so guessing 'legit' every time doesn't win."""
    d = store.get(split)
    model = _model(d, model)
    rng = np.random.default_rng(seed)
    s, t = d.models[model].scores, d.models_meta[model]["threshold"]
    r = rng.random()
    if r < 0.45:
        pool = np.flatnonzero(d.y == 1)
    elif r < 0.75:
        pool = None                                       # random legit, sampled below
    else:
        pool = np.flatnonzero((d.y == 0) & (s >= t))
    if pool is None:
        while True:
            i = int(rng.integers(len(d.y)))
            if d.y[i] == 0:
                break
    else:
        i = int(rng.choice(pool))
    rows = _card_rows(d.split, int(d.tx["user"].iat[i]), int(d.tx["card"].iat[i]))
    pos = int(np.flatnonzero(rows == i)[0])
    hist = rows[max(0, pos - 6):pos]
    return {"round": int(d.tx["txn_id"].iat[i]), "transaction": _records(d, np.array([i]))[0],
            "history": _records(d, hist[::-1])}


@router.get("/game/reveal/{txn_id}")
def game_reveal(txn_id: int, split: str | None = None):
    d = store.get(split)
    if txn_id not in d.row_of_txn.index:
        raise HTTPException(404, "unknown transaction")
    i = int(d.row_of_txn[txn_id])
    scores = [{"model": k, "score": float(d.models[k].scores[i]), "threshold": d.models_meta[k]["threshold"],
               "flagged": bool(d.models[k].scores[i] >= d.models_meta[k]["threshold"])} for k in d.model_keys]
    reasons = None
    ex = d.explainable()
    if ex:
        e = d.explain(ex[0], txn_id, top=4)
        reasons = {"model": ex[0], "items": e["contributions"]}
    return {"txn_id": txn_id, "is_fraud": int(d.y[i]), "scores": scores, "reasons": reasons}


# ----------------------------------------------------------------------------- card attack timelines
@router.get("/cards/attacks")
def card_attacks(split: str | None = None, model: str | None = None, limit: int = Query(40, le=200)):
    """Cards with the most fraud in the split: candidates for an attack replay."""
    d = store.get(split)
    model = _model(d, model)
    f = np.flatnonzero(d.y == 1)
    df = pd.DataFrame({"user": d.tx["user"].to_numpy()[f], "card": d.tx["card"].to_numpy()[f],
                       "ts": d.ts_sec[f], "amount": np.abs(d.amount[f]),
                       "caught": d.models[model].scores[f] >= d.models_meta[model]["threshold"],
                       "city": d.tx["city"].to_numpy()[f]})
    g = df.groupby(["user", "card"]).agg(frauds=("ts", "size"), first=("ts", "min"), last=("ts", "max"),
                                         fraud_amount=("amount", "sum"), caught=("caught", "sum"),
                                         city=("city", lambda c: c.mode().iat[0]))
    g = g.sort_values(["frauds", "fraud_amount"], ascending=False).head(limit).reset_index()
    g["caught"] = g["caught"].astype(int)
    return {"model": model, "cards": g.to_dict(orient="records")}


@router.get("/cards/{user}/{card}")
def card_timeline(user: int, card: int, split: str | None = None, model: str | None = None):
    d = store.get(split)
    model = _model(d, model)
    rows = _card_rows(d.split, user, card)
    s = d.models[model].scores[rows]
    recs = _records(d, rows, {"score": s, "is_fraud": d.y[rows]})
    fts = d.ts_sec[rows][d.y[rows] == 1]
    window = [int(fts.min()) - 21 * 86400, int(fts.max()) + 10 * 86400] if len(fts) else None
    return {"model": model, "threshold": d.models_meta[model]["threshold"], "user": user, "card": card,
            "rows": recs, "attack_window": window}


# ----------------------------------------------------------------------------- triage
def _triage_queue(d: SplitData, model: str, date: str, k: int) -> np.ndarray:
    day = int(pd.Timestamp(date).timestamp())
    lo, hi = np.searchsorted(d.ts_sec, [day, day + 86400])
    if hi <= lo:
        raise HTTPException(404, "no transactions that day")
    s = d.models[model].scores[lo:hi]
    top = np.argsort(-s, kind="stable")[:k]
    return lo + top


@router.get("/triage")
def triage(date: str, split: str | None = None, model: str | None = None, k: int = Query(24, ge=5, le=60)):
    """The day's k riskiest transactions by model score, labels withheld."""
    d = store.get(split)
    model = _model(d, model)
    q = _triage_queue(d, model, date, k)
    day = int(pd.Timestamp(date).timestamp())
    lo, hi = np.searchsorted(d.ts_sec, [day, day + 86400])
    return {"model": model, "date": date, "threshold": d.models_meta[model]["threshold"],
            "day_transactions": int(hi - lo), "queue": _records(d, q, {"score": d.models[model].scores[q]})}


@router.get("/triage/reveal")
def triage_reveal(date: str, split: str | None = None, model: str | None = None, k: int = Query(24, ge=5, le=60)):
    d = store.get(split)
    model = _model(d, model)
    q = _triage_queue(d, model, date, k)
    day = int(pd.Timestamp(date).timestamp())
    lo, hi = np.searchsorted(d.ts_sec, [day, day + 86400])
    yy = d.y[lo:hi]
    return {"labels": {int(d.tx["txn_id"].iat[i]): int(d.y[i]) for i in q},
            "day_frauds": int(yy.sum()),
            "day_fraud_amount": float(np.abs(d.amount[lo:hi][yy == 1]).sum())}


@router.get("/triage/days")
def triage_days(split: str | None = None, n: int = 12, seed: int = 0):
    """Some interesting days (with several frauds) to suggest."""
    d = store.get(split)
    days = d.ts_sec[d.y == 1] // 86400
    u, c = np.unique(days, return_counts=True)
    good = u[c >= 4]
    pick = np.sort(np.random.default_rng(seed).choice(good, size=min(n, len(good)), replace=False))
    return {"days": [str(np.datetime64(int(x), "D")) for x in pick]}
