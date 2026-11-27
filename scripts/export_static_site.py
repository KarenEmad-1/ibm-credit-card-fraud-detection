"""Export the dashboard as a static site for GitHub Pages: python scripts/export_static_site.py <out-dir>

GitHub Pages serves files and runs no Python, so this writes what
dashboard/js/static-backend.js needs to answer the dashboard's requests in the browser:

    index.html, js/, styles.css, vendor/   the dashboard, marked as static
    data/splits.json               the splits served (the TEST/VAL switch), as /api/splits
    data/<split>/static.json       results that need Python: metrics, curves, segments, attack lists
    data/<split>/core.bin.gz       transaction columns, little-endian arrays
    data/<split>/score-<i>.bin.gz  one float32 score column per model, loaded when needed
    data/<split>/shap/<n>.json     SHAP explanations for a chosen set of transactions

<split> is every split the API has loaded (FRAUD_SPLITS): test, and val when the sample
bundle has it.

Everything is read through the API's own store and endpoint functions, so the site shows
what the API shows. Nothing is trained and no split is re-scored. Runs on a sample bundle
only (the full year is too large for a browser), with the same environment variables as
the API. CI runs it inside the tested sample image.
"""
from __future__ import annotations

import argparse
import gzip
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi.encoders import jsonable_encoder  # noqa: E402

from api.config import settings  # noqa: E402
from api.main import FRONTEND  # noqa: E402
from api.routers import performance, play  # noqa: E402
from api.store import store  # noqa: E402

SHAP_SHARDS = 64
GAME_LEGIT = 1500           # ordinary legit transactions the game can draw
GAME_FALSE_ALERTS = 150     # per model
ATTACK_CARDS = 40           # as requested by the CASES screen
TRIAGE_DAYS = {"n": 14, "seed": 7}   # as requested by the TRIAGE screen
PAGE_FILES = ("index.html", "styles.css", "js", "vendor")
META_TAG = '  <meta name="fraud-ops-backend" content="static">\n'


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(jsonable_encoder(obj), separators=(",", ":"), allow_nan=False), encoding="utf-8")


def write_gz(path: Path, data: bytes) -> None:
    path.write_bytes(gzip.compress(data, compresslevel=9, mtime=0))    # mtime=0: same input, same file


def core_columns(d) -> tuple[list[dict], dict, bytes]:
    """(column list, dictionaries for text columns, the arrays back to back)."""
    arrays: dict[str, np.ndarray] = {
        "txn_id": d.tx["txn_id"].to_numpy(np.int32), "ts": d.ts_sec.astype(np.int32),
        "user": d.tx["user"].to_numpy(np.int32), "amount": d.tx["amount"].to_numpy(np.float32),
        "mcc": d.tx["mcc"].to_numpy(np.int16), "card": d.tx["card"].to_numpy(np.int16),
    }
    dictionaries = {}
    for name in ("merchant", "city", "state", "channel", "errors"):
        col = d.tx[name].astype("category").cat
        if (col.codes < 0).any():
            raise SystemExit(f"column {name} has missing values")
        dictionaries[name] = [str(v) for v in col.categories]
        small = len(dictionaries[name]) <= 256
        arrays[name] = col.codes.to_numpy().astype(np.uint8 if small else np.uint16)
    arrays["is_fraud"] = d.y.astype(np.uint8)

    columns, blob = [], b""
    for name, a in sorted(arrays.items(), key=lambda kv: -kv[1].itemsize):   # widest first: offsets stay aligned
        columns.append({"name": name, "type": f"{a.dtype.kind}{a.dtype.itemsize}", "offset": len(blob)})
        blob += a.astype(a.dtype.newbyteorder("<")).tobytes()
    return columns, dictionaries, blob


def game_pools(d) -> dict:
    """Transactions the game can draw. All of them get a SHAP explanation."""
    rng = np.random.default_rng(0)
    ids = d.tx["txn_id"].to_numpy()

    def sample(mask: np.ndarray, n: int) -> list[int]:
        idx = np.flatnonzero(mask)
        return ids[np.sort(rng.choice(idx, size=min(n, len(idx)), replace=False))].tolist()

    return {"fraud": ids[d.y == 1].tolist(), "legit": sample(d.y == 0, GAME_LEGIT),
            "false_alerts": {k: sample((d.y == 0) & (d.models[k].scores >= d.models_meta[k]["threshold"]),
                                       GAME_FALSE_ALERTS) for k in d.model_keys}}


def export_split(d, data: Path) -> int:
    """Write one split's files into `data`; return how many transactions have SHAP."""
    split = d.split
    (data / "shap").mkdir(parents=True)
    # 2. Transaction and score columns
    columns, dictionaries, blob = core_columns(d)
    write_gz(data / "core.bin.gz", blob)
    for i, key in enumerate(d.model_keys):
        write_gz(data / f"score-{i}.bin.gz", d.tx[f"score::{key}"].to_numpy(np.float32).astype("<f4").tobytes())

    # 3. SHAP for every transaction the CASES and HUNT screens can ask about
    shap_model = d.explainable()[0]
    attacks = {k: play.card_attacks(split, k, ATTACK_CARDS) for k in d.model_keys}
    ids = d.tx["txn_id"].to_numpy()
    pools = game_pools(d)
    cards = {(c["user"], c["card"]) for r in attacks.values() for c in r["cards"]}
    wanted = set(pools["fraud"]) | set(pools["legit"]) | {t for p in pools["false_alerts"].values() for t in p}
    for user, card in cards:
        wanted |= set(ids[play._card_rows(split, user, card)].tolist())
    explained = d.explain_many(shap_model, sorted(wanted))
    features = sorted({c["feature"] for e in explained for c in e["contributions"]})
    code = {f: i for i, f in enumerate(features)}
    shards: list[dict] = [{} for _ in range(SHAP_SHARDS)]
    for e in explained:
        shards[e["txn_id"] % SHAP_SHARDS][e["txn_id"]] = [
            round(e["base_logit"], 6), round(e["logit"], 6),
            e["other_features"]["count"], round(e["other_features"]["shap"], 6),
            [[code[c["feature"]], c["value"], round(c["shap"], 6)] for c in e["contributions"]]]
    for n, shard in enumerate(shards):
        write_json(data / "shap" / f"{n}.json", shard)

    # 4. Everything else Python computes, straight from the endpoint functions
    write_json(data / "static.json", {
        "n_rows": len(d.tx), "columns": columns, "dictionaries": dictionaries,
        "models": performance.models(split), "curves": performance.curves(split),
        "drift": performance.drift(split), "ablations": performance.ablations(),
        "segments": {k: performance.segments(k, split, None) for k in d.model_keys},
        "attacks": attacks, "triage_days": play.triage_days(split, **TRIAGE_DAYS), "game": pools,
        "shap": {"model": shap_model, "shards": SHAP_SHARDS, "features": features},
    })
    return len(wanted)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    args = ap.parse_args()
    out: Path = args.out
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty")

    store.load()
    for d in store.splits.values():
        if "sample" not in d.meta:
            raise SystemExit(f"the static site needs a sample bundle; split {d.split} is the full one")

    # 1. The dashboard
    out.mkdir(parents=True, exist_ok=True)
    for name in PAGE_FILES:
        src = FRONTEND / name
        if src.is_dir():
            shutil.copytree(src, out / name)
        elif src.exists():
            shutil.copy(src, out / name)
    page = out / "index.html"
    html = page.read_text(encoding="utf-8")
    if html.count("</head>") != 1:
        raise SystemExit("index.html: expected one </head>")
    page.write_text(html.replace("</head>", META_TAG + "</head>"), encoding="utf-8", newline="\n")

    data = out / "data"
    data.mkdir()
    write_json(data / "splits.json", performance.splits())
    summary = []
    for split, d in store.splits.items():
        n_explained = export_split(d, data / split)
        summary.append(f"{split}: {len(d.tx):,} transactions, {n_explained:,} explained")

    # 5. Licence files for the derived data (they sit next to the bundle's dashboard/ folder)
    for name in ("NOTICE", "LICENSE-IBM-DATA.txt"):
        shutil.copy(settings.dashboard_dir.parent / name, data / name)

    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"static site in {out}: {'; '.join(summary)}; {size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
