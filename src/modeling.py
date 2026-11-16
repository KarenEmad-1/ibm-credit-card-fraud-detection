"""Helpers shared by the feature-selection and model notebooks."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.data import CACHE_DIR
from src import features as F

TARGET = "is_fraud"


def load_split(name: str, columns: list[str] | None = None) -> pd.DataFrame:
    """Load a Phase 1 feature table ('train', 'val' or 'test')."""
    cols = None if columns is None else list(dict.fromkeys([*columns, TARGET]))
    df = pd.read_parquet(CACHE_DIR / f"features_{name}.parquet", columns=cols)
    for c in categorical_in(list(df.columns)):       # guard against Parquet dtype round-trips
        if not isinstance(df[c].dtype, pd.CategoricalDtype):
            df[c] = df[c].astype(str).astype("category")
    return df


def feature_sets() -> dict:
    return json.loads((CACHE_DIR / "feature_sets.json").read_text())


def undersample(df: pd.DataFrame, neg_per_pos: float, seed: int = 0) -> pd.DataFrame:
    """Keep every fraud and a random sample of non-frauds (`neg_per_pos` per fraud).

    Only ever applied to the training period. Evaluation always uses the
    original class balance, because PR-AUC depends on it.
    """
    pos = df[df[TARGET] == 1]
    neg = df[df[TARGET] == 0]
    n_neg = min(len(neg), int(len(pos) * neg_per_pos))
    out = pd.concat([pos, neg.sample(n=n_neg, random_state=seed)])
    return out.sample(frac=1, random_state=seed)        # shuffle


def categorical_in(features: list[str]) -> list[str]:
    return [c for c in features if c in F.CATEGORICAL_FEATURES]


def _cats_to_str(X: pd.DataFrame) -> pd.DataFrame:
    return X.astype(object).where(X.notna(), "missing").astype(str)


def linear_preprocessor(features: list[str]) -> ColumnTransformer:
    """For linear models / neural nets: numeric -> median impute (+ missing flag) and
    scale; categorical -> one-hot (rare levels pooled, unseen levels ignored)."""
    cats = categorical_in(features)
    nums = [c for c in features if c not in cats]
    return ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                          ("scale", StandardScaler())]), nums),
        ("cat", Pipeline([("to_str", FunctionTransformer(_cats_to_str, feature_names_out="one-to-one")),
                          ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                                   min_frequency=200, sparse_output=False,
                                                   dtype=np.float32))]), cats),
    ], verbose_feature_names_out=False)


def catboost_frame(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """CatBoost wants categorical columns as strings (no NaN)."""
    X = df[features].copy()
    for c in categorical_in(features):
        X[c] = _cats_to_str(X[[c]])[c]
    return X


def tree_matrix(df: pd.DataFrame, features: list[str]) -> np.ndarray:
    """For sklearn tree models: categories -> integer codes (levels already aligned
    to train in Phase 1), missing -> -1. Trees don't need scaling or one-hot."""
    X = df[features].copy()
    for c in categorical_in(features):
        X[c] = X[c].cat.codes
    return X.fillna(-1).to_numpy(np.float32)


def source_feature(encoded_name: str, features: list[str]) -> str:
    """Map an encoded column ('use_chip_Swipe Transaction', 'missingindicator_x')
    back to the feature it came from."""
    if encoded_name.startswith("missingindicator_"):
        return encoded_name[len("missingindicator_"):]
    if encoded_name in features:
        return encoded_name
    for f in sorted(features, key=len, reverse=True):
        if encoded_name.startswith(f + "_"):
            return f
    return encoded_name
