"""Model comparison, threshold simulator, segments and ablations."""
from __future__ import annotations

import re

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from api.store import REPORTS_DIR, store
from src.data import PROJECT_ROOT

router = APIRouter(prefix="/api", tags=["performance"])


def _model(split: str | None, key: str):
    d = store.get(split)
    if key not in d.models:
        raise HTTPException(404, f"unknown model '{key}'. Known: {d.model_keys}")
    return d


@router.get("/splits")
def splits():
    return {"default": store.default_split,
            "splits": [{"split": s, "year": d.meta["year"], "built_at": d.meta["built_at"],
                        "n_transactions": d.meta["n_transactions"], "n_frauds": d.meta["n_frauds"]}
                       for s, d in store.splits.items()]}


@router.get("/models")
def models(split: str | None = None):
    d = store.get(split)
    ci = None
    path = REPORTS_DIR / "bootstrap_ci.csv"
    if d.split == "test" and path.exists():
        ci = pd.read_csv(path).to_dict(orient="records")
    return {"split": d.split, "year": d.meta["year"], "n_transactions": d.meta["n_transactions"],
            "n_frauds": d.meta["n_frauds"], "n_days": d.meta["n_days"],
            "models": d.meta["models"], "explainable": d.explainable(), "bootstrap_ci": ci}


@router.get("/curves")
def curves(split: str | None = None):
    d = store.get(split)
    return {"base_rate": d.meta["n_frauds"] / d.meta["n_transactions"], "curves": d.meta["curves"]}


@router.get("/drift")
def drift(split: str | None = None):
    """Daily PSI of key features vs training 2015-2017 (trailing 7-day window)."""
    return store.get(split).meta["drift"]


@router.get("/threshold/{key}")
def threshold(key: str, split: str | None = None,
              t: float | None = Query(None, ge=0, le=1, description="decision threshold"),
              alert_rate: float | None = Query(None, gt=0, le=1, description="share of transactions to flag")):
    d = _model(split, key)
    if t is None:
        t = d.models[key].threshold_for_rate(alert_rate) if alert_rate else d.models_meta[key]["threshold"]
    return {"model": key, "val_threshold": d.models_meta[key]["threshold"], **d.at_threshold(key, t)}


@router.get("/segments/{key}")
def segments(key: str, split: str | None = None, t: float | None = Query(None, ge=0, le=1)):
    d = _model(split, key)
    t = d.models_meta[key]["threshold"] if t is None else t
    return {"model": key, "threshold": t, "rows": d.segments(key, t)}


@router.get("/ablations")
def ablations():
    """Notebook 07 results: whatever it has written so far."""
    tables = {}
    for name, file in [("catboost_variants", "val_metrics_ablation_catboost.csv"),
                       ("variant_b_all_models", "val_metrics_variant_B.csv"),
                       ("geo_leakage_probe", "geo_leakage_probe.csv")]:
        path = REPORTS_DIR / file
        if path.exists():
            df = pd.read_csv(path)
            df = df.rename(columns={df.columns[0]: "name"}) if df.columns[0].startswith("Unnamed") else df
            tables[name] = df.where(df.notna(), None).to_dict(orient="records")

    # progress log, so the page shows results while notebook 07 is still running
    log_rows = []
    log = PROJECT_ROOT / "models" / "ablation_run.log"
    if log.exists():
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
            m = re.search(r"(\S.*?) ([A-D]): val PR-AUC ([0-9.]+)", line[20:])
            if m:
                log_rows.append({"model": m.group(1), "variant": m.group(2), "pr_auc": float(m.group(3))})
    latest = {(r["model"], r["variant"]): r for r in log_rows}       # last run of each wins
    return {"complete": "variant_b_all_models" in tables, "tables": tables,
            "log": list(latest.values()),
            "variants": {"A": "30 selected features (includes is_international, zip_freq)",
                         "B": "A without geography (is_international, zip_freq)",
                         "C": "A + merchant_state (the suspected shortcut)",
                         "D": "A + user snapshot features (age, income, FICO...)"}}
