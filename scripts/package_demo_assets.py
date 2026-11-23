"""Package what the dashboard serves into one versioned archive for a GitHub Release.

    python scripts/package_demo_assets.py                 # -> dist/fraud-ops-demo-<version>.tar.gz (+ .sha256)
    python scripts/package_demo_assets.py --version v2
    python scripts/package_demo_assets.py --sample-start 2019-01-01 --sample-end 2019-02-28
        -> dist/fraud-ops-demo-sample-<version>.tar.gz, for hosts with ~512 MB of memory

The Docker build downloads this archive and checks it against the SHA-256 pinned
in the Dockerfile, so the image is built from exactly these bytes.

Nothing here trains, tunes or re-scores a model, and no file is resampled: the
2019 test year is shipped whole, so every number the demo shows is the one in
the README. The archive holds only *derived* files, never the raw 24.4M-row CSV:

  dashboard/test/transactions.parquet   1.72M test-year rows: display columns + the saved
                                        scores of all 8 models (from notebook 08)
  dashboard/test/meta.json              precomputed PR curves, metrics, drift
  features/features_test.parquet        the 2019 feature table, restricted to the columns
                                        the two served CatBoost models use (for SHAP)
  models/catboost_tuned.cbm             CatBoost A (scored on test in notebook 08)
  models/final/catboost_B.joblib        CatBoost B, the headline model
  models/ablation_run.log               notebook 07 progress log (SHORTCUT screen)
  NOTICE, LICENSE-IBM-DATA.txt          attribution and Apache-2.0 terms of the source data
  MANIFEST.json                         SHA-256 and size of every file, source commit

Run it after scripts/build_dashboard_data.py --split test.

Sample mode (--sample-start/--sample-end) keeps one contiguous window of the test year,
so the replay, card timelines and triage days stay realistic. The precomputed headline
metrics and PR curves in meta.json stay the FULL-year ones (meta["full_year"] holds the
full-year counts); n_transactions/n_frauds/n_days describe the rows actually served, so
everything computed live (threshold simulator, segments, triage) is labelled as sample.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE = PROJECT_ROOT / "data" / "cache"
MODELS = PROJECT_ROOT / "models"
DIST = PROJECT_ROOT / "dist"

NOTICE = """\
FRAUD-OPS demo data bundle
==========================

This archive contains files DERIVED from the "Credit Card Transactions" synthetic
dataset created by IBM (TabFormer project, https://github.com/IBM/TabFormer;
Kaggle mirror https://www.kaggle.com/datasets/ealtman2019/credit-card-transactions),
which is distributed under the Apache License, Version 2.0. A copy of that licence
is included as LICENSE-IBM-DATA.txt.

Changes made (Apache-2.0 section 4(b)): the original transaction records were
cleaned, restricted to the year 2019, reduced to a subset of columns, joined with
engineered features, and extended with fraud scores produced by models trained in
https://github.com/KarenEmad-1/ibm-credit-card-fraud-detection. The files are not the
original dataset and are not endorsed by IBM.

The data is synthetic: it describes no real person, card or merchant.

The model files and code-generated outputs in this archive are released under the MIT
licence of the repository above.
"""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def write_sample(start: str, end: str, out_dir: Path):
    """Write one contiguous window of the test year; return the kept txn_ids, sorted."""
    tx = pd.read_parquet(CACHE / "dashboard" / "test" / "transactions.parquet")
    lo, hi = pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta(days=1)
    keep = tx[(tx["ts"] >= lo) & (tx["ts"] < hi)]
    if keep.empty:
        sys.exit(f"no transactions between {start} and {end}")
    keep.to_parquet(out_dir / "transactions.parquet", index=False)

    meta = json.loads((CACHE / "dashboard" / "test" / "meta.json").read_text())
    meta["full_year"] = {k: meta[k] for k in ("n_transactions", "n_frauds", "n_days")}
    meta.update(n_transactions=int(len(keep)), n_frauds=int(keep["is_fraud"].sum()),
                n_days=int(keep["ts"].dt.normalize().nunique()))
    meta["sample"] = {
        "start": start, "end": end,
        "note": (f"Sample: {start} to {end} of the 2019 test year. Headline metrics and PR curves "
                 "are the full-year results; screens computed live use only the sample."),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    print(f"sample {start}..{end}: {meta['n_transactions']:,} of {meta['full_year']['n_transactions']:,} "
          f"transactions, {meta['n_frauds']} of {meta['full_year']['n_frauds']} frauds, {meta['n_days']} days")
    return np.sort(keep["txn_id"].to_numpy())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", default="v1")
    ap.add_argument("--sample-start", help="first day kept, e.g. 2019-01-01 (sample mode)")
    ap.add_argument("--sample-end", help="last day kept, inclusive, e.g. 2019-02-28 (sample mode)")
    args = ap.parse_args()
    sample = bool(args.sample_start or args.sample_end)
    if sample and not (args.sample_start and args.sample_end):
        sys.exit("--sample-start and --sample-end go together")

    name = f"fraud-ops-demo-{'sample-' if sample else ''}{args.version}"
    stage = DIST / name
    if stage.exists():
        shutil.rmtree(stage)
    (stage / "dashboard" / "test").mkdir(parents=True)
    (stage / "features").mkdir()
    (stage / "models" / "final").mkdir(parents=True)

    required = [CACHE / "dashboard" / "test" / "transactions.parquet", CACHE / "dashboard" / "test" / "meta.json",
                CACHE / "features_test.parquet", MODELS / "catboost_tuned.cbm",
                MODELS / "final" / "catboost_B.joblib", MODELS / "ablation_run.log"]
    missing = [str(p.relative_to(PROJECT_ROOT)) for p in required if not p.exists()]
    if missing:
        sys.exit(f"Missing inputs: {missing}. Run notebooks 07-08 and build_dashboard_data.py --split test first.")

    if sample:
        rows = write_sample(args.sample_start, args.sample_end, stage / "dashboard" / "test")
    else:
        for src in required[:2]:
            shutil.copy2(src, stage / "dashboard" / "test" / src.name)
    shutil.copy2(MODELS / "catboost_tuned.cbm", stage / "models" / "catboost_tuned.cbm")
    shutil.copy2(MODELS / "final" / "catboost_B.joblib", stage / "models" / "final" / "catboost_B.joblib")
    shutil.copy2(MODELS / "ablation_run.log", stage / "models" / "ablation_run.log")

    # Only the feature columns the served models read. Row order is kept: the API looks
    # rows up by position (txn_id = row number in features_test.parquet).
    model_a = CatBoostClassifier()
    model_a.load_model(str(MODELS / "catboost_tuned.cbm"))
    model_b = joblib.load(MODELS / "final" / "catboost_B.joblib")
    cols = sorted(set(model_a.feature_names_) | set(model_b["features"]))
    feats = pd.read_parquet(CACHE / "features_test.parquet", columns=cols)
    n_full = len(pd.read_parquet(CACHE / "dashboard" / "test" / "transactions.parquet", columns=["txn_id"]))
    if len(feats) != n_full:
        sys.exit(f"features_test has {len(feats)} rows but transactions has {n_full}: rebuild the dashboard data")
    if sample:
        # Keep the sampled rows and say which transaction each one is (the API then
        # looks rows up by txn_id instead of by position)
        feats = feats.iloc[rows].reset_index(drop=True)
        feats.insert(0, "txn_id", rows.astype("int32"))
    n_tx = len(feats)
    feats.to_parquet(stage / "features" / "features_test.parquet", index=False, compression="zstd")

    (stage / "NOTICE").write_text(NOTICE, encoding="utf-8")
    shutil.copy2(PROJECT_ROOT / "licenses" / "IBM-TabFormer-LICENSE.txt", stage / "LICENSE-IBM-DATA.txt")

    files = sorted(p for p in stage.rglob("*") if p.is_file())
    manifest = {
        "bundle": name, "source_commit": git_commit(),
        "split": (f"test (2019), SAMPLE {args.sample_start} to {args.sample_end}" if sample
                  else "test (2019, complete)"),
        "n_transactions": n_tx, "shap_feature_columns": cols,
        "files": {str(p.relative_to(stage)).replace("\\", "/"): {"bytes": p.stat().st_size, "sha256": sha256(p)}
                  for p in files},
    }
    (stage / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    archive = DIST / f"{name}.tar.gz"
    # Sorted names, fixed tar metadata and a zero gzip timestamp: the same inputs give
    # the same archive bytes, so the checksum identifies the content, not the build time.
    with open(archive, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as gz,             tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for p in sorted(stage.rglob("*")):
            info = tar.gettarinfo(str(p), arcname=str(p.relative_to(DIST)).replace("\\", "/"))
            info.mtime, info.uid, info.gid, info.uname, info.gname = 0, 0, 0, "", ""
            info.mode = 0o755 if p.is_dir() else 0o644
            if p.is_file():
                with p.open("rb") as f:
                    tar.addfile(info, f)
            else:
                tar.addfile(info)
    digest = sha256(archive)
    # LF even on Windows: `sha256sum -c` reads a trailing CR as part of the file name
    (DIST / f"{name}.tar.gz.sha256").write_text(f"{digest}  {archive.name}\n", encoding="utf-8", newline="\n")
    shutil.rmtree(stage)

    print(f"{archive.relative_to(PROJECT_ROOT)}  {archive.stat().st_size / 2**20:.1f} MiB")
    print(f"sha256  {digest}")
    for k, v in manifest["files"].items():
        print(f"  {v['bytes'] / 2**20:8.1f} MiB  {k}")


if __name__ == "__main__":
    main()
