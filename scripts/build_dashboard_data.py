"""Build the data the dashboard serves, from scores the notebooks already saved.

    python scripts/build_dashboard_data.py --split val    # validation year 2018
    python scripts/build_dashboard_data.py --split test   # test year 2019, from notebook 08's saved scores

Nothing here trains, tunes or re-scores a model. The script only gathers saved
scores, joins them to the transaction details the dashboard displays, and
precomputes the heavy parts (PR curves, metrics, daily drift):

- val  : the tuned models' 2018 scores from notebooks 04-06 (variant A), plus the
         variant B-D scores from notebook 07 once `val_scores_ablation.parquet` exists,
         plus the MLP's A and B scores from notebook 09 (validation only: the MLP was
         added after the test year was scored, so it never appears in the test split).
- test : `test_scores_final.parquet`, written once by notebook 08. The 2019 test year
         is never read before that file exists, so the dashboard cannot peek at test.

Decision thresholds are always the ones chosen on validation.

Output: data/cache/dashboard/<split>/{transactions.parquet, meta.json}
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import CACHE_DIR                                   # noqa: E402
from src.evaluation import evaluate, best_f1_threshold           # noqa: E402

REPORTS = PROJECT_ROOT / "reports"
OUT_ROOT = CACHE_DIR / "dashboard"
YEARS = {"val": 2018, "test": 2019}
MODEL_ORDER = ["LogReg", "RandomForest", "CatBoost", "FT-Transformer", "MLP"]

# Columns shown in the transaction tables and used for segments.
FEATURE_COLS = ["amount", "is_online", "is_international", "is_new_merchant_for_card",
                "has_error", "hour", "card_txns_24h", "is_fraud"]
RAW_COLS = ["User", "Card", "Transaction Datetime", "Merchant Name", "Merchant City",
            "Merchant State", "MCC", "Use Chip", "Errors?"]

# Drift: compared with a recent training reference (2015-2017).
DRIFT_FEATURES = ["amount_signed_log", "secs_since_prev_txn", "card_txns_24h",
                  "amt_card_zscore", "merchant_freq", "is_online"]
DRIFT_REF_START = 2015
DRIFT_WINDOW_DAYS = 7


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')}  {msg}", flush=True)


# ----------------------------------------------------------------------------- scores
def _family(key: str) -> str:
    return next(m for m in MODEL_ORDER if key.startswith(m))


def gather_scores(split: str) -> tuple[pd.DataFrame, dict[str, float | None]]:
    """Return (scores with one column per model key like 'CatBoost A', val thresholds)."""
    variants_file = REPORTS / "feature_variants.json"
    val_thresholds = json.loads(variants_file.read_text())["val_thresholds"] if variants_file.exists() else {}

    if split == "test":
        path = CACHE_DIR / "test_scores_final.parquet"
        if not path.exists():
            sys.exit("test_scores_final.parquet not found: run notebook 08 first. "
                     "The 2019 test year is scored once there, never here.")
        scores = pd.read_parquet(path)
        missing = [k for k in scores if k not in val_thresholds]
        if missing:
            sys.exit(f"No validation threshold for {missing} in feature_variants.json")
        return scores, {k: val_thresholds[k] for k in scores}

    # validation: variant A = the tuned models of notebooks 04-06
    cols = {}
    thresholds = {}
    for file, metrics_file, names in [
        ("val_scores_traditional.parquet", "val_metrics_traditional.csv", ["LogReg", "RandomForest"]),
        ("val_scores_catboost.parquet", "val_metrics_catboost.csv", ["CatBoost"]),
        ("val_scores_ft_transformer.parquet", "val_metrics_ft_transformer.csv", ["FT-Transformer"]),
        ("val_scores_mlp.parquet", "val_metrics_mlp.csv", ["MLP"]),
    ]:
        if not (CACHE_DIR / file).exists():
            log(f"  skip {file}: not found")
            continue
        df = pd.read_parquet(CACHE_DIR / file)
        metrics = pd.read_csv(REPORTS / metrics_file, index_col=0)
        for n in names:
            if n == "MLP":                                     # notebook 09 saves "MLP A" / "MLP B"
                for v in ("A", "B"):
                    cols[f"MLP {v}"] = df[f"MLP {v}"]
                    thresholds[f"MLP {v}"] = float(metrics.loc["MLP (tuned)" if v == "A" else "MLP B", "threshold"])
                continue
            cols[f"{n} A"] = df[f"{n} (tuned)"]
            thresholds[f"{n} A"] = float(metrics.loc[f"{n} (tuned)", "threshold"])

    # variants B-D from notebook 07, when it has finished
    ablation = CACHE_DIR / "val_scores_ablation.parquet"
    if ablation.exists():
        df = pd.read_parquet(ablation)
        for k in df.columns:
            if k.split()[-1] in ("B", "C", "D") and k not in cols:
                cols[k] = df[k]
                thresholds[k] = val_thresholds.get(k)          # None -> chosen on val below
    else:
        log("  val_scores_ablation.parquet not found yet (notebook 07): variant A only")
    return pd.DataFrame(cols), thresholds


def sort_keys(keys) -> list[str]:
    return sorted(keys, key=lambda k: (k.split()[-1], MODEL_ORDER.index(_family(k))))


# ----------------------------------------------------------------------------- curves
def pr_curve_points(y: np.ndarray, p: np.ndarray, n: int = 300) -> dict:
    prec, rec, thr = precision_recall_curve(y, p)
    prec, rec = prec[:-1], rec[:-1]                     # drop the (1, 0) end point
    # even spacing in recall, so the curve looks right without a million points
    grid = np.linspace(rec.min(), rec.max(), n)
    idx = np.unique(np.clip(np.searchsorted(-rec, -grid), 0, len(rec) - 1))
    return {"precision": prec[idx].round(5).tolist(), "recall": rec[idx].round(5).tolist(),
            "threshold": thr[idx].round(6).tolist()}


# ----------------------------------------------------------------------------- drift
def _bin_edges(ref: pd.Series) -> np.ndarray:
    vals = ref.dropna().to_numpy(np.float64)
    if np.unique(vals).size <= 10:                      # binary / small discrete: one bin per value
        u = np.unique(vals)
        return np.r_[-np.inf, (u[:-1] + u[1:]) / 2, np.inf]
    q = np.unique(np.quantile(vals, np.linspace(0, 1, 11)[1:-1]))
    return np.r_[-np.inf, q, np.inf]


def _hist(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Share per bin, with NaN as its own last bin."""
    nan = np.isnan(x)
    counts = np.r_[np.histogram(x[~nan], edges)[0], nan.sum()].astype(np.float64)
    return counts / max(counts.sum(), 1)


def _psi(expected: np.ndarray, actual: np.ndarray, eps: float = 1e-4) -> float:
    e, a = np.clip(expected, eps, None), np.clip(actual, eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


def daily_drift(feats: pd.DataFrame, days: np.ndarray) -> dict:
    log(f"  drift reference: training {DRIFT_REF_START}-2017")
    ref = pd.read_parquet(CACHE_DIR / "features_train.parquet", columns=DRIFT_FEATURES + ["year"],
                          filters=[("year", ">=", DRIFT_REF_START)])
    edges = {f: _bin_edges(ref[f]) for f in DRIFT_FEATURES}
    expected = {f: _hist(ref[f].to_numpy(np.float64), edges[f]) for f in DRIFT_FEATURES}
    del ref

    uniq = np.unique(days)
    out = {}
    for d in uniq:
        lo = d - np.timedelta64(DRIFT_WINDOW_DAYS - 1, "D")
        m = (days >= lo) & (days <= d)
        out[str(d)] = {f: round(_psi(expected[f], _hist(feats.loc[m, f].to_numpy(np.float64), edges[f])), 5)
                       for f in DRIFT_FEATURES}
    return {"features": DRIFT_FEATURES, "window_days": DRIFT_WINDOW_DAYS,
            "reference": f"train {DRIFT_REF_START}-2017", "daily": out}


# ----------------------------------------------------------------------------- main
def build(split: str) -> None:
    year = YEARS[split]
    out_dir = OUT_ROOT / split
    out_dir.mkdir(parents=True, exist_ok=True)

    log(f"[{split} {year}] gathering saved scores")
    scores, thresholds = gather_scores(split)
    keys = sort_keys(scores.columns)
    log(f"  models: {keys}")

    log("  loading features and transaction details")
    feats = pd.read_parquet(CACHE_DIR / f"features_{split}.parquet",
                            columns=sorted(set(FEATURE_COLS) | set(DRIFT_FEATURES)))
    raw = pd.read_parquet(CACHE_DIR / "transactions_clean.parquet", columns=RAW_COLS + ["Year", "Amount"],
                          filters=[("Year", "==", year)])
    assert len(raw) == len(feats) == len(scores), (len(raw), len(feats), len(scores))
    # the feature tables keep the raw file's row order; check it before joining by position
    assert np.allclose(raw["Amount"].to_numpy(np.float64), feats["amount"].to_numpy(np.float64)), \
        "row order of features and raw transactions differs"

    y = feats["is_fraud"].to_numpy()
    metrics, curves = {}, {}
    for k in keys:
        p = scores[k].to_numpy(np.float64)
        if thresholds.get(k) is None:                    # only possible on validation
            thresholds[k] = best_f1_threshold(y, p)
        metrics[k] = {m: float(v) for m, v in evaluate(y, p, threshold=thresholds[k]).items()}
        curves[k] = pr_curve_points(y, p)
        log(f"  {k}: PR-AUC {metrics[k]['pr_auc']:.4f}, threshold {thresholds[k]:.4f}")

    ts = raw["Transaction Datetime"]
    tx = pd.DataFrame({
        "txn_id": np.arange(len(raw), dtype=np.int32),   # row position in features_<split>.parquet
        "ts": ts.to_numpy(),
        "user": raw["User"].astype(np.int32).to_numpy(),
        "card": raw["Card"].astype(np.int16).to_numpy(),
        "amount": feats["amount"].astype(np.float32).to_numpy(),
        "merchant": raw["Merchant Name"].astype(str).str[-6:].to_numpy(),
        "city": raw["Merchant City"].astype(str).to_numpy(),
        "state": raw["Merchant State"].astype(str).replace("nan", "").to_numpy(),
        "mcc": raw["MCC"].astype(np.int16).to_numpy(),
        "channel": raw["Use Chip"].astype(str).str.replace(" Transaction", "").to_numpy(),
        "errors": raw["Errors?"].astype(str).replace("nan", "").to_numpy(),
        "is_online": feats["is_online"].astype(np.int8).to_numpy(),
        "is_international": feats["is_international"].astype(np.int8).to_numpy(),
        "is_new_merchant": feats["is_new_merchant_for_card"].astype(np.int8).to_numpy(),
        "has_error": feats["has_error"].astype(np.int8).to_numpy(),
        "hour": feats["hour"].astype(np.int8).to_numpy(),
        "is_fraud": y.astype(np.int8),
    })
    for k in keys:
        tx[f"score::{k}"] = scores[k].astype(np.float32).to_numpy()

    order = np.lexsort((tx["txn_id"].to_numpy(), tx["ts"].to_numpy()))   # chronological, stable
    tx = tx.iloc[order].reset_index(drop=True)
    tx.to_parquet(out_dir / "transactions.parquet", index=False)

    log("  daily drift")
    days = ts.to_numpy().astype("datetime64[D]")
    drift = daily_drift(feats, days)

    meta = {
        "split": split, "year": year, "built_at": time.strftime("%Y-%m-%d %H:%M"),
        "n_transactions": int(len(tx)), "n_frauds": int(y.sum()),
        "n_days": int(np.unique(days).size),
        "models": [{"key": k, "family": _family(k), "variant": k.split()[-1],
                    "threshold": float(thresholds[k]), "metrics": metrics[k]} for k in keys],
        "curves": curves,
        "drift": drift,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta))
    log(f"  wrote {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], default="val")
    build(ap.parse_args().split)
