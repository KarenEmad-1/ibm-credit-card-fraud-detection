"""Runtime configuration, read once from environment variables.

Every setting has a default that matches the repo layout, so a local
`uvicorn api.main:app` works with no configuration. The Docker image overrides
the paths to point at the demo bundle (see scripts/package_demo_assets.py).

| Variable              | Default                          | What it points at                          |
|-----------------------|----------------------------------|--------------------------------------------|
| FRAUD_DASHBOARD_DIR   | data/cache/dashboard             | <split>/transactions.parquet + meta.json   |
| FRAUD_FEATURES_DIR    | data/cache                       | features_<split>.parquet (SHAP inputs)     |
| FRAUD_MODELS_DIR      | models                           | catboost_tuned.cbm, final/catboost_*.joblib |
| FRAUD_REPORTS_DIR     | reports                          | result CSVs shown on the ablation screens  |
| FRAUD_ABLATION_LOG    | models/ablation_run.log          | notebook 07 progress log                   |
| FRAUD_SPLITS          | test,val                         | splits to load, in preference order        |
| HOST                  | 127.0.0.1                        | bind address (the image sets 0.0.0.0)      |
| PORT                  | 8000                             | port (hosts such as Render set $PORT)      |
| LOG_LEVEL             | info                             | debug / info / warning / error             |
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _path(var: str, default: Path) -> Path:
    value = os.environ.get(var)
    return Path(value).expanduser().resolve() if value else default


@dataclass(frozen=True)
class Settings:
    dashboard_dir: Path
    features_dir: Path
    models_dir: Path
    reports_dir: Path
    ablation_log: Path
    splits: tuple[str, ...]
    host: str
    port: int
    log_level: str

    @classmethod
    def from_env(cls) -> "Settings":
        models_dir = _path("FRAUD_MODELS_DIR", PROJECT_ROOT / "models")
        splits = tuple(s.strip() for s in os.environ.get("FRAUD_SPLITS", "test,val").split(",") if s.strip())
        unknown = set(splits) - {"test", "val"}
        if unknown or not splits:
            raise ValueError(f"FRAUD_SPLITS must list 'test' and/or 'val', got {os.environ.get('FRAUD_SPLITS')!r}")
        return cls(
            dashboard_dir=_path("FRAUD_DASHBOARD_DIR", PROJECT_ROOT / "data" / "cache" / "dashboard"),
            features_dir=_path("FRAUD_FEATURES_DIR", PROJECT_ROOT / "data" / "cache"),
            models_dir=models_dir,
            reports_dir=_path("FRAUD_REPORTS_DIR", PROJECT_ROOT / "reports"),
            ablation_log=_path("FRAUD_ABLATION_LOG", models_dir / "ablation_run.log"),
            splits=splits,
            host=os.environ.get("HOST", "127.0.0.1"),
            port=int(os.environ.get("PORT", "8000")),
            log_level=os.environ.get("LOG_LEVEL", "info").lower(),
        )


settings = Settings.from_env()
