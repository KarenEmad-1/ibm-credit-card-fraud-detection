"""One evaluation protocol for every model, so comparisons are fair.

- PR-AUC (average precision) is the headline metric: fraud is 0.1-0.15% of
  transactions, and ROC-AUC looks great for almost any model at that imbalance.
- Recall at a fixed precision: "of the frauds, how many do we catch while at
  least half of our alerts are real fraud?"
- Precision/recall at an alert budget: investigators can review a fixed number
  of alerts; we flag the top 0.1% riskiest transactions.
- The decision threshold is chosen on VALIDATION (max F1) and then applied
  unchanged to test, never re-chosen on test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

ALERT_RATE = 0.001          # review budget: top 0.1% of transactions
TARGET_PRECISION = 0.5


def recall_at_precision(y, p, min_precision=TARGET_PRECISION) -> float:
    prec, rec, _ = precision_recall_curve(y, p)
    ok = prec >= min_precision
    return float(rec[ok].max()) if ok.any() else 0.0


def at_alert_budget(y, p, rate=ALERT_RATE) -> tuple[float, float]:
    """Precision and recall when flagging the top `rate` share of transactions."""
    y = np.asarray(y)
    k = max(1, int(round(len(y) * rate)))
    top = np.argpartition(-np.asarray(p), k - 1)[:k]
    hits = y[top].sum()
    return float(hits / k), float(hits / max(y.sum(), 1))


def best_f1_threshold(y, p) -> float:
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    return float(thr[np.nanargmax(f1)])


def evaluate(y, p, threshold: float | None = None) -> dict:
    """All metrics for one set of scores. Pass the validation-chosen threshold
    when evaluating test; leave it None on validation to choose it there."""
    y = np.asarray(y)
    p = np.asarray(p)
    if threshold is None:
        threshold = best_f1_threshold(y, p)
    pred = p >= threshold
    tp = int((pred & (y == 1)).sum())
    prec_budget, rec_budget = at_alert_budget(y, p)
    return {
        "pr_auc": average_precision_score(y, p),
        "roc_auc": roc_auc_score(y, p),
        f"recall@precision{TARGET_PRECISION:g}": recall_at_precision(y, p),
        "precision@top0.1%": prec_budget,
        "recall@top0.1%": rec_budget,
        "threshold": threshold,
        "precision@threshold": tp / max(int(pred.sum()), 1),
        "recall@threshold": tp / max(int(y.sum()), 1),
        "alerts@threshold": int(pred.sum()),
    }


def compare(results: dict[str, dict]) -> pd.DataFrame:
    return pd.DataFrame(results).T.round(4)


def plot_pr_curves(y, scores: dict[str, np.ndarray], ax=None, title="Precision-recall (validation 2018)"):
    import matplotlib.pyplot as plt
    ax = ax or plt.subplots(figsize=(8, 6))[1]
    for name, p in scores.items():
        prec, rec, _ = precision_recall_curve(y, p)
        ax.plot(rec, prec, label=f"{name} (PR-AUC {average_precision_score(y, p):.3f})")
    ax.axhline(np.mean(y), color="gray", ls=":", label=f"random ({np.mean(y):.4f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend()
    return ax
