"""Frozen final recipes for the five models.

The ablations (notebook 07) and the single test evaluation (notebook 08) must train
and score each model exactly as notebooks 04-06 did, only on a different feature
list. These functions do that. The MLP (notebook 09) was added after the test year
was scored, so it is fitted and scored on validation only. Hyperparameters are read from
`reports/best_params_*.json` and nothing here is tuned.

Each `fit_*` returns a "bundle" dict that `score(bundle, df)` turns into fraud
probabilities. `save_bundle` / `load_bundle` store bundles in `models/final/`.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data import PROJECT_ROOT
from src.modeling import (TARGET, undersample, linear_preprocessor, tree_matrix,
                          catboost_frame, categorical_in)

SEED = 0
REPORTS_DIR = PROJECT_ROOT / "reports"
FINAL_DIR = PROJECT_ROOT / "models" / "final"
# The four models of notebooks 04-08, the only ones ever scored on the 2019 test year.
# Notebooks 07 and 08 loop over this list, so the MLP must never be added to it.
MODEL_NAMES = ["LogReg", "RandomForest", "CatBoost", "FT-Transformer"]
VAL_ONLY_MODELS = ["MLP"]            # notebook 09: added after the test year was scored


def load_params() -> dict:
    trad = json.loads((REPORTS_DIR / "best_params_traditional.json").read_text())
    cb = json.loads((REPORTS_DIR / "best_params_catboost.json").read_text())["catboost"]
    ft = json.loads((REPORTS_DIR / "best_params_ft_transformer.json").read_text())["ft_transformer"]
    mlp_path = REPORTS_DIR / "best_params_mlp.json"
    mlp = json.loads(mlp_path.read_text())["mlp"] if mlp_path.exists() else None
    return {
        "LogReg": trad["logreg"]["params"],
        "RandomForest": trad["random_forest"]["params"],
        "CatBoost": cb["params"],
        "FT-Transformer": {**ft["params"], "seeds": ft["seeds"], "start_year": ft["start_year"]},
        **({"MLP": {**mlp["params"], "seeds": mlp["seeds"], "start_year": mlp["start_year"]}} if mlp else {}),
    }


def _sample(train: pd.DataFrame, p: dict) -> pd.DataFrame:
    """The same undersample as notebooks 04-06 (same rows for the same train frame)."""
    return undersample(train[train["year"] >= p["start_year"]], p["neg_per_pos"], seed=SEED)


# ----------------------------------------------------------------------------- fitting
def fit_logreg(train, features, p):
    from sklearn.linear_model import LogisticRegression
    s = _sample(train, p)
    pre = linear_preprocessor(features)
    X = pre.fit_transform(s[features])
    if p["penalty"] == "l1":
        clf = LogisticRegression(l1_ratio=1, solver="liblinear", C=p["C"], class_weight=p["class_weight"], max_iter=1000)
    else:
        clf = LogisticRegression(l1_ratio=0, solver="lbfgs", C=p["C"], class_weight=p["class_weight"], max_iter=2000)
    clf.fit(X, s[TARGET])
    return {"kind": "LogReg", "features": features, "preprocessor": pre, "model": clf}


def fit_rf(train, features, p, n_estimators=500):
    from sklearn.ensemble import RandomForestClassifier
    s = _sample(train, p)
    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=p["max_depth"],
                                min_samples_leaf=p["min_samples_leaf"], max_features=p["max_features"],
                                class_weight=p["class_weight"], n_jobs=-1, random_state=SEED)
    rf.fit(tree_matrix(s, features), s[TARGET])
    return {"kind": "RandomForest", "features": features, "model": rf}


def fit_catboost(train, features, p):
    from catboost import CatBoostClassifier, Pool
    s = _sample(train, p)
    cats = categorical_in(features)
    params = {k: v for k, v in p.items() if k not in ("start_year", "neg_per_pos")}
    model = CatBoostClassifier(**params, random_seed=SEED, verbose=0, thread_count=-1, task_type="CPU",
                               train_dir=str(PROJECT_ROOT / "models" / "catboost_info"))
    model.fit(Pool(catboost_frame(s, features), s[TARGET], cat_features=cats))
    return {"kind": "CatBoost", "features": features, "model": model}


def fit_ft_transformer(train, features, p, ckpt_dir, tag, val=None, verbose=True, kind="FT-Transformer"):
    """3-seed stabilised FT-Transformer (notebook 06's final recipe), or with
    kind="MLP" the MLP of notebook 09 trained the same way. `val`, if given,
    is only recorded per epoch (select_best=False), never used to pick anything."""
    import torch
    from src.ft_transformer import TabularPreprocessor, to_device, fit_ft, checkpoint_name, MODEL_KEYS
    from src.mlp import MLP, MLP_KEYS
    if kind == "MLP":
        MODEL_KEYS, model_fn = MLP_KEYS, MLP
    else:
        model_fn = None
    s = _sample(train, p)
    pre = TabularPreprocessor(features, seed=SEED).fit(s)
    tr = to_device(*pre.transform(s), s[TARGET])
    va = to_device(*pre.transform(val), val[TARGET]) if val is not None else None
    train_keys = ("epochs", "schedule", "warmup_epochs", "ema_decay", "select_best", "lr",
                  "weight_decay", "batch_size")
    kw = {k: p[k] for k in train_keys} | {k: p[k] for k in MODEL_KEYS if k in p}
    states, histories = [], []
    for seed in p["seeds"]:
        cfg = {**kw, "neg_per_pos": p["neg_per_pos"], "start_year": p["start_year"],
               "features": sorted(features), "seed": seed}
        m, h = fit_ft(tr, pre.n_num, pre.cardinalities, va, seed=seed, verbose=verbose, model_fn=model_fn,
                      checkpoint=Path(ckpt_dir) / checkpoint_name(tag, cfg), **kw)
        states.append({k: v.detach().cpu() for k, v in m.state_dict().items()})
        histories.append(h)
        del m
    del tr, va
    torch.cuda.empty_cache()
    model_params = {k: p[k] for k in MODEL_KEYS if k in p}
    return {"kind": kind, "features": features, "preprocessor": pre,
            "state_dicts": states, "model_params": model_params, "histories": histories}


def fit(name, train, features, params, **kw):
    p = params[name]
    if name == "LogReg":
        return fit_logreg(train, features, p)
    if name == "RandomForest":
        return fit_rf(train, features, p)
    if name == "CatBoost":
        return fit_catboost(train, features, p)
    if name == "FT-Transformer":
        return fit_ft_transformer(train, features, p, **kw)
    if name == "MLP":
        return fit_ft_transformer(train, features, p, kind="MLP", **kw)
    raise ValueError(name)


# ----------------------------------------------------------------------------- scoring
def score(bundle, df: pd.DataFrame) -> np.ndarray:
    kind, features = bundle["kind"], bundle["features"]
    if kind == "LogReg":
        return bundle["model"].predict_proba(bundle["preprocessor"].transform(df[features]))[:, 1]
    if kind == "RandomForest":
        return bundle["model"].predict_proba(tree_matrix(df, features))[:, 1]
    if kind == "CatBoost":
        from catboost import Pool
        return bundle["model"].predict_proba(Pool(catboost_frame(df, features),
                                                  cat_features=categorical_in(features)))[:, 1]
    if kind in ("FT-Transformer", "MLP"):
        import torch
        from src.ft_transformer import FTTransformer, to_device, predict
        from src.mlp import MLP
        net = MLP if kind == "MLP" else FTTransformer
        pre = bundle["preprocessor"]
        x_num, x_cat = to_device(*pre.transform(df))
        out = []
        for sd in bundle["state_dicts"]:
            m = net(pre.n_num, pre.cardinalities, **bundle["model_params"]).cuda()
            m.load_state_dict(sd)
            out.append(predict(m, x_num, x_cat))
            del m
        del x_num, x_cat
        torch.cuda.empty_cache()
        return np.mean(out, axis=0)                   # average of the seeds' probabilities
    raise ValueError(kind)


# ----------------------------------------------------------------------------- storage
def bundle_path(name: str, variant: str) -> Path:
    return FINAL_DIR / f"{name.lower().replace('-', '_')}_{variant}.joblib"


def save_bundle(bundle, name, variant) -> None:
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    path = bundle_path(name, variant)
    tmp = path.with_suffix(".tmp")
    joblib.dump(bundle, tmp, compress=3)
    tmp.replace(path)                                 # atomic: a power cut can't leave half a file


def load_bundle(name, variant):
    path = bundle_path(name, variant)
    return joblib.load(path) if path.exists() else None


def load_variant_a(name: str, features: list[str]):
    """Variant A = the tuned models already saved by notebooks 04-06 and 09."""
    import torch
    models = PROJECT_ROOT / "models"
    if name == "LogReg":
        b = joblib.load(models / "logreg_tuned.joblib")
        return {"kind": name, "features": b["features"], "preprocessor": b["preprocessor"], "model": b["model"]}
    if name == "RandomForest":
        b = joblib.load(models / "random_forest_tuned.joblib")
        return {"kind": name, "features": b["features"], "model": b["model"]}
    if name == "CatBoost":
        from catboost import CatBoostClassifier
        m = CatBoostClassifier(); m.load_model(str(models / "catboost_tuned.cbm"))
        return {"kind": name, "features": features, "model": m}
    if name == "FT-Transformer":
        from src.ft_transformer import MODEL_KEYS
        b = torch.load(models / "ft_transformer_tuned.pt", map_location="cpu", weights_only=False)
        return {"kind": name, "features": b["preprocessor"].features, "preprocessor": b["preprocessor"],
                "state_dicts": b["state_dicts"], "model_params": {k: b["params"][k] for k in MODEL_KEYS if k in b["params"]}}
    if name == "MLP":
        from src.mlp import MLP_KEYS
        b = torch.load(models / "mlp_tuned.pt", map_location="cpu", weights_only=False)
        return {"kind": name, "features": b["preprocessor"].features, "preprocessor": b["preprocessor"],
                "state_dicts": b["state_dicts"], "model_params": {k: b["params"][k] for k in MLP_KEYS if k in b["params"]}}
    raise ValueError(name)
