"""Smoke-test a running FRAUD-OPS server: python scripts/smoke_test.py <base-url> [--expect-vendored]

Used by CI against the freshly built image, and against the live demo URL after a
deploy. Standard library only, so it runs anywhere. Exits non-zero on the first
failed check and prints what it measured.

Checks:
  1. /health answers and has the test split loaded
  2. the headline number is intact: CatBoost B test PR-AUC 0.4808, 2,083 alerts
     (on a sample bundle: PR-AUC is still the full-year value; the alert count is
     computed live on the sample, so only the sample label is checked)
  3. SHAP explanation for a transaction reproduces its saved score
  4. the alert-triage simulator returns a queue for a suggested day
  5. the replay stream (server-sent events) delivers its first events promptly,
     including through whatever proxy sits in front of the app
  6. (--expect-vendored) the page loads Preact/D3 from the image, not a CDN
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from urllib.parse import quote

HEADLINE = "CatBoost B"
EXPECTED_PR_AUC = 0.4808          # README: CatBoost B, 2019 test year
EXPECTED_ALERTS = 2083
SSE_FIRST_EVENT_LIMIT_S = 10.0    # generous: a free host may be slow, but it must stream


def fail(msg: str) -> None:
    print(f"FAIL  {msg}")
    sys.exit(1)


def ok(msg: str) -> None:
    print(f"ok    {msg}")


def get_json(base: str, path: str, timeout: float = 60) -> dict:
    t0 = time.perf_counter()
    with urllib.request.urlopen(base + path, timeout=timeout) as r:
        if r.status != 200:
            fail(f"{path} -> HTTP {r.status}")
        body = json.loads(r.read())
    get_json.last_ms = (time.perf_counter() - t0) * 1000
    return body


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("base_url")
    ap.add_argument("--expect-vendored", action="store_true", help="page must not load scripts from a CDN")
    ap.add_argument("--wait", type=float, default=0, help="seconds to keep retrying /health (cold starts)")
    args = ap.parse_args()
    base = args.base_url.rstrip("/")
    key = quote(HEADLINE)

    # 1. health, retrying while the server (or a sleeping free-tier host) starts
    deadline, t0 = time.time() + args.wait, time.perf_counter()
    while True:
        try:
            h = get_json(base, "/health", timeout=30)
            break
        except Exception as e:                       # noqa: BLE001 - any failure means "not up yet"
            if time.time() > deadline:
                fail(f"/health not reachable: {e}")
            time.sleep(2)
    if h.get("status") != "ok" or "test" not in h.get("splits", []):
        fail(f"/health unexpected: {h}")
    ok(f"/health ok after {time.perf_counter() - t0:.1f}s: {h}")

    # 2. headline metrics, straight from the saved evaluation
    models = get_json(base, "/api/models")
    cb = next((m for m in models["models"] if m["key"] == HEADLINE), None)
    if cb is None:
        fail(f"{HEADLINE} missing from /api/models")
    pr_auc = cb["metrics"]["pr_auc"]
    if round(pr_auc, 4) != EXPECTED_PR_AUC:
        fail(f"{HEADLINE} PR-AUC {pr_auc} != {EXPECTED_PR_AUC}")
    thr = get_json(base, f"/api/threshold/{key}")
    sample = models.get("sample")
    if sample:
        full = models["full_year"]
        if not (0 < models["n_transactions"] < full["n_transactions"]) or thr["alerts"] <= 0:
            fail(f"sample bundle inconsistent: {models['n_transactions']} of {full['n_transactions']}, "
                 f"{thr['alerts']} alerts")
        ok(f"SAMPLE bundle {sample['start']}..{sample['end']}: {models['n_transactions']:,} of "
           f"{full['n_transactions']:,} transactions; full-year {HEADLINE} PR-AUC {pr_auc:.4f}; "
           f"{thr['alerts']} alerts on the sample")
    else:
        if thr["alerts"] != EXPECTED_ALERTS:
            fail(f"{HEADLINE} alerts at validation threshold {thr['alerts']} != {EXPECTED_ALERTS}")
        ok(f"{HEADLINE}: PR-AUC {pr_auc:.4f}, {thr['alerts']} alerts, precision {thr['precision']:.3f}, "
           f"recall {thr['recall']:.3f}")

    # 3. SHAP reproduces the saved score of a known fraud
    fraud = get_json(base, f"/api/transactions?model={key}&kind=caught&limit=1&seed=0")["rows"][0]
    exp = get_json(base, f"/api/explain/{key}/{fraud['txn_id']}")
    shap_ms = get_json.last_ms
    if abs(exp["probability"] - fraud["score"]) > 1e-5:
        fail(f"SHAP probability {exp['probability']} != saved score {fraud['score']}")
    top = exp["contributions"][0]
    ok(f"SHAP txn {fraud['txn_id']}: p={exp['probability']:.4f} matches saved score; "
       f"top feature {top['feature']} ({top['shap']:+.2f}); {shap_ms:.0f} ms")

    # 4. triage simulator
    days = get_json(base, "/api/triage/days")["days"]
    day = days[0]["date"] if isinstance(days[0], dict) else days[0]
    queue = get_json(base, f"/api/triage?date={day}")["queue"]
    if not queue:
        fail(f"empty triage queue for {day}")
    ok(f"triage {day}: {len(queue)} alerts in the queue")

    # 5. server-sent events arrive promptly (a buffering proxy would hold them back)
    t0 = time.perf_counter()
    req = urllib.request.Request(f"{base}/api/replay?model={key}&speed=600",
                                 headers={"Accept": "text/event-stream"})
    events, first = [], None
    with urllib.request.urlopen(req, timeout=SSE_FIRST_EVENT_LIMIT_S + 5) as r:
        ctype = r.headers.get("Content-Type", "")
        if not ctype.startswith("text/event-stream"):
            fail(f"replay Content-Type {ctype!r}")
        for line in r:
            if line.startswith(b"event:"):
                events.append(line.split(b":", 1)[1].strip().decode())
                first = first or time.perf_counter() - t0
            if len(events) >= 5 or time.perf_counter() - t0 > SSE_FIRST_EVENT_LIMIT_S:
                break
    elapsed = time.perf_counter() - t0
    if not events or first > SSE_FIRST_EVENT_LIMIT_S or "tick" not in events:
        fail(f"replay stream: events {events} after {elapsed:.1f}s")
    ok(f"replay SSE: first event after {first * 1000:.0f} ms, {len(events)} events in {elapsed:.1f}s {events}")

    # 6. no CDN in the image's page
    if args.expect_vendored:
        with urllib.request.urlopen(base + "/", timeout=30) as r:
            html = r.read().decode()
        if "unpkg.com" in html or "jsdelivr" in html or "fonts.googleapis" in html:
            fail("index.html still references a CDN")
        ok("index.html loads Preact, D3 and fonts from the image")

    print("all checks passed")


if __name__ == "__main__":
    main()
