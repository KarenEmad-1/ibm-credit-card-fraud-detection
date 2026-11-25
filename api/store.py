"""In-memory data for the dashboard, loaded once at startup.

Everything comes from `<FRAUD_DASHBOARD_DIR>/<split>/` (default
`data/cache/dashboard/<split>/`), built by `scripts/build_dashboard_data.py`.
Paths are set in `api/config.py`. The API never re-scores a whole split: metrics at
any threshold come from sorted saved scores, so each request is O(log n).
"""
from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score

from api.config import settings

log = logging.getLogger("fraud_ops.store")

DASHBOARD_DIR = settings.dashboard_dir
REPORTS_DIR = settings.reports_dir
TEXT_COLUMNS = ["merchant", "city", "state", "channel", "errors"]
SPLIT_ORDER = list(settings.splits)               # test is the headline once notebook 08 has run


@dataclass
class ModelScores:
    key: str
    scores: np.ndarray                             # chronological order, like the transactions
    sorted_desc: np.ndarray = field(init=False)
    cum_tp: np.ndarray = field(init=False)         # frauds among the top k
    cum_fraud_amt: np.ndarray = field(init=False)  # fraud dollars among the top k

    def init(self, y: np.ndarray, amount: np.ndarray) -> "ModelScores":
        order = np.argsort(-self.scores, kind="stable")
        self.sorted_desc = self.scores[order]
        fraud = y[order]
        self.cum_tp = np.r_[0, np.cumsum(fraud)]
        self.cum_fraud_amt = np.r_[0.0, np.cumsum(np.where(fraud == 1, np.abs(amount[order]), 0.0))]
        return self

    def n_flagged(self, t: float) -> int:
        """Transactions with score >= t (ties all included)."""
        return int(np.searchsorted(-self.sorted_desc, -t, side="right"))

    def threshold_for_rate(self, rate: float) -> float:
        k = int(np.clip(round(len(self.scores) * rate), 1, len(self.scores)))
        return float(self.sorted_desc[k - 1])


class SplitData:
    def __init__(self, split: str):
        d = DASHBOARD_DIR / split
        self.split = split
        self.meta = json.loads((d / "meta.json").read_text())
        # Repeated text values are read straight into category codes (Arrow dictionary
        # encoding), so the 1.7M-row Python strings are never created: same values, a
        # fraction of the memory. Those strings were ~80% of the frame, and Python keeps
        # the memory of freed small objects, so converting after loading wouldn't help.
        path = d / "transactions.parquet"
        text_cols = [c for c in TEXT_COLUMNS if c in pq.read_schema(path).names]
        self.tx = pq.read_table(path, read_dictionary=text_cols).to_pandas()
        self.model_keys = [m["key"] for m in self.meta["models"]]
        self.models_meta = {m["key"]: m for m in self.meta["models"]}
        self.y = self.tx["is_fraud"].to_numpy().astype(np.int64)
        self.amount = self.tx["amount"].to_numpy(np.float64)
        self.ts_sec = self.tx["ts"].to_numpy().astype("datetime64[s]").astype(np.int64)
        self.models = {k: ModelScores(k, self.tx[f"score::{k}"].to_numpy(np.float64)).init(self.y, self.amount)
                       for k in self.model_keys}
        self.row_of_txn = pd.Series(np.arange(len(self.tx)), index=self.tx["txn_id"].to_numpy())
        self._features: pd.DataFrame | None = None
        self._explainers: dict = {}
        self._segment_ap: dict = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ metrics
    def at_threshold(self, key: str, t: float) -> dict:
        m = self.models[key]
        n, frauds = len(self.y), int(self.y.sum())
        k = m.n_flagged(t)
        tp = int(m.cum_tp[k])
        caught_amt = float(m.cum_fraud_amt[k])
        total_amt = float(m.cum_fraud_amt[-1])
        days = self.meta["n_days"]
        return {
            "threshold": t, "alerts": k, "alerts_per_day": k / days, "alert_rate": k / n,
            "tp": tp, "fp": k - tp, "fn": frauds - tp, "tn": n - k - (frauds - tp),
            "precision": tp / k if k else 0.0, "recall": tp / frauds if frauds else 0.0,
            "fraud_amount_caught": caught_amt, "fraud_amount_total": total_amt,
        }

    SEGMENTS = {
        "Channel": lambda tx: tx["channel"].astype(str),
        "Merchant location": lambda tx: np.where(tx["is_international"] == 1, "International",
                                                 np.where(tx["is_online"] == 1, "Online", "Domestic in-store")),
        "First purchase at merchant": lambda tx: np.where(tx["is_new_merchant"] == 1, "New merchant", "Seen before"),
        "Transaction error": lambda tx: np.where(tx["has_error"] == 1, "Had an error", "No error"),
        "Amount": lambda tx: pd.cut(tx["amount"].abs(), [-1, 10, 100, 500, np.inf],
                                    labels=["< $10", "$10-100", "$100-500", "> $500"]).astype(str),
        "Time of day": lambda tx: pd.cut(tx["hour"], [-1, 5, 11, 17, 23],
                                         labels=["Night 0-5", "Morning 6-11", "Afternoon 12-17", "Evening 18-23"]).astype(str),
    }

    @cached_property
    def segment_labels(self) -> dict[str, np.ndarray]:
        # Kept as categoricals: six cached text arrays of 1.7M labels cost ~580 MiB
        return {name: pd.Categorical(np.asarray(fn(self.tx))) for name, fn in self.SEGMENTS.items()}

    def segments(self, key: str, t: float) -> list[dict]:
        s = self.models[key].scores
        flagged = s >= t
        out = []
        for name, labels in self.segment_labels.items():
            for level in pd.unique(labels):
                m = labels == level
                y = self.y[m]
                n_fraud = int(y.sum())
                ck = (key, name, level)
                if ck not in self._segment_ap:
                    self._segment_ap[ck] = float(average_precision_score(y, s[m])) if 0 < n_fraud < m.sum() else None
                f = flagged[m]
                tp = int((f & (y == 1)).sum())
                out.append({"segment": name, "level": str(level), "transactions": int(m.sum()),
                            "frauds": n_fraud, "fraud_rate": n_fraud / max(int(m.sum()), 1),
                            "share_of_all_fraud": n_fraud / max(int(self.y.sum()), 1),
                            "pr_auc": self._segment_ap[ck], "alerts": int(f.sum()),
                            "precision": tp / int(f.sum()) if f.sum() else None,
                            "recall": tp / n_fraud if n_fraud else None})
        return out

    # ------------------------------------------------------------------ explanations
    def features(self) -> pd.DataFrame:
        with self._lock:
            if self._features is None:
                from src.modeling import categorical_in
                cols = sorted({c for e in self._explainer_specs().values() for c in e["features"]})
                # Same as src.modeling.load_split, but from the configured directory
                path = settings.features_dir / f"features_{self.split}.parquet"
                # A sample bundle stores only some rows, with a txn_id column saying which;
                # the full table has one row per transaction, in txn_id order.
                keyed = "txn_id" in pq.read_schema(path).names
                df = pd.read_parquet(path, columns=cols + ["txn_id"] if keyed else cols)
                if keyed:
                    df = df.set_index("txn_id")
                for c in categorical_in(list(df.columns)):
                    if not isinstance(df[c].dtype, pd.CategoricalDtype):
                        df[c] = df[c].astype(str).astype("category")
                self._features = df
            return self._features

    def _explainer_specs(self) -> dict:
        """CatBoost models that can be explained with SHAP, keyed like the scores."""
        if "specs" in self._explainers:
            return self._explainers["specs"]
        import joblib
        from catboost import CatBoostClassifier
        specs = {}
        for key in self.model_keys:
            fam, variant = self.models_meta[key]["family"], self.models_meta[key]["variant"]
            if fam != "CatBoost":
                continue
            if variant == "A":
                path = settings.models_dir / "catboost_tuned.cbm"
                if path.exists():
                    m = CatBoostClassifier()
                    m.load_model(str(path))
                    specs[key] = {"model": m, "features": list(m.feature_names_)}
            else:
                path = settings.models_dir / "final" / f"catboost_{variant}.joblib"
                if path.exists():
                    b = joblib.load(path)
                    specs[key] = {"model": b["model"], "features": list(b["features"])}
        self._explainers["specs"] = specs
        return specs

    def explainable(self) -> list[str]:
        return list(self._explainer_specs())

    def explain(self, key: str, txn_id: int, top: int = 12) -> dict:
        return self.explain_many(key, [txn_id], top)[0]

    def explain_many(self, key: str, txn_ids: list[int], top: int = 12) -> list[dict]:
        from catboost import Pool
        from src.modeling import catboost_frame, categorical_in
        spec = self._explainer_specs()[key]
        feats = spec["features"]
        table = self.features()
        rows = table.loc[txn_ids] if table.index.name == "txn_id" else table.iloc[txn_ids]
        X = catboost_frame(rows, feats)
        values = spec["model"].get_feature_importance(Pool(X, cat_features=categorical_in(feats)), type="ShapValues")
        out = []
        for n, (txn_id, shap) in enumerate(zip(txn_ids, values, strict=True)):
            base, contrib = float(shap[-1]), shap[:-1]
            order = np.argsort(-np.abs(contrib))
            items = [{"feature": feats[i], "value": _jsonable(X.iloc[n, i]), "shap": float(contrib[i])}
                     for i in order[:top]]
            rest = float(contrib[order[top:]].sum()) if len(order) > top else 0.0
            logit = base + float(contrib.sum())
            out.append({"model": key, "txn_id": txn_id, "base_logit": base, "logit": logit,
                        "probability": float(1 / (1 + np.exp(-logit))), "contributions": items,
                        "other_features": {"count": max(len(order) - top, 0), "shap": rest}})
        return out


def _jsonable(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return None if np.isnan(v) else round(float(v), 4)
    return str(v)


class Store:
    def __init__(self):
        self.splits: dict[str, SplitData] = {}

    def load(self) -> None:
        for s in SPLIT_ORDER:
            if (DASHBOARD_DIR / s / "meta.json").exists():
                t0 = time.perf_counter()
                self.splits[s] = SplitData(s)
                log.info("loaded split %s: %d transactions in %.1fs", s, len(self.splits[s].tx),
                         time.perf_counter() - t0)
        # Arrow's allocator keeps freed read buffers for reuse; hand them back to the OS
        import pyarrow as pa
        pa.default_memory_pool().release_unused()
        if not self.splits:
            raise RuntimeError(f"No dashboard data found in {DASHBOARD_DIR}. "
                               "Run: python scripts/build_dashboard_data.py --split val")

    @property
    def default_split(self) -> str:
        return next(s for s in SPLIT_ORDER if s in self.splits)

    def get(self, split: str | None) -> SplitData:
        from fastapi import HTTPException
        split = split or self.default_split
        if split not in self.splits:
            raise HTTPException(404, f"split '{split}' not built")
        return self.splits[split]


store = Store()
