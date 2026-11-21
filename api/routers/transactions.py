"""Browse individual transactions and explain a model's score with SHAP."""
from __future__ import annotations

import numpy as np
from fastapi import APIRouter, HTTPException, Query

from api.store import store

router = APIRouter(prefix="/api", tags=["transactions"])

DISPLAY = ["txn_id", "ts", "user", "card", "amount", "merchant", "city", "state", "mcc",
           "channel", "errors", "is_fraud"]
KINDS = {"caught", "missed", "false_alert", "fraud", "top", "random"}


def _rows(d, idx, key):
    df = d.tx.iloc[idx][DISPLAY].copy()
    df["ts"] = df["ts"].dt.strftime("%Y-%m-%d %H:%M")
    df["score"] = d.models[key].scores[idx]
    df["flagged"] = df["score"] >= d.models_meta[key]["threshold"]
    return df.to_dict(orient="records")


@router.get("/transactions")
def list_transactions(model: str, kind: str = "caught", split: str | None = None,
                      limit: int = Query(15, ge=1, le=100), seed: int = 0):
    d = store.get(split)
    if model not in d.models:
        raise HTTPException(404, f"unknown model '{model}'")
    if kind not in KINDS:
        raise HTTPException(400, f"kind must be one of {sorted(KINDS)}")
    s, t, y = d.models[model].scores, d.models_meta[model]["threshold"], d.y
    if kind == "top":
        idx = np.argsort(-s)[:limit]
    else:
        mask = {"caught": (s >= t) & (y == 1), "missed": (s < t) & (y == 1),
                "false_alert": (s >= t) & (y == 0), "fraud": y == 1,
                "random": np.ones_like(y, dtype=bool)}[kind]
        pool = np.flatnonzero(mask)
        idx = np.sort(np.random.default_rng(seed).choice(pool, size=min(limit, len(pool)), replace=False)) \
            if len(pool) else pool
    return {"model": model, "kind": kind, "threshold": t, "rows": _rows(d, idx, model)}


@router.get("/transactions/{txn_id}")
def transaction(txn_id: int, split: str | None = None):
    d = store.get(split)
    if txn_id not in d.row_of_txn.index:
        raise HTTPException(404, "unknown transaction")
    i = int(d.row_of_txn[txn_id])
    row = d.tx.iloc[i][DISPLAY].to_dict()
    row["ts"] = row["ts"].strftime("%Y-%m-%d %H:%M")
    row = {k: (v.item() if hasattr(v, "item") else v) for k, v in row.items()}
    scores = [{"model": k, "score": float(d.models[k].scores[i]),
               "threshold": d.models_meta[k]["threshold"],
               "flagged": bool(d.models[k].scores[i] >= d.models_meta[k]["threshold"])} for k in d.model_keys]
    return {"transaction": row, "scores": scores}


@router.get("/explain/{model}/{txn_id}")
def explain(model: str, txn_id: int, split: str | None = None):
    d = store.get(split)
    if model not in d.explainable():
        raise HTTPException(400, f"SHAP explanations are served for CatBoost models only: {d.explainable()}")
    if txn_id not in d.row_of_txn.index:
        raise HTTPException(404, "unknown transaction")
    return d.explain(model, txn_id)
