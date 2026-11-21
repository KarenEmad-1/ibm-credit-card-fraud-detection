"""Live replay: stream a split's transactions in time order as if they were arriving now.

The scores are the saved ones (no model runs per event), so this is a playback of
what the model would have flagged, not new scoring. Sent as server-sent events:

- `tick`: transactions in the last slice of simulated time, plus the alerts among them
- `day` : summary of a finished day (volume, alerts, true/false alerts, missed frauds, drift)
- `end` : the split is over

Fraud labels are included with each event; the client hides them until the
configured label delay has passed in simulated time, as in production.
"""
from __future__ import annotations

import asyncio
import json

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from api.store import store

router = APIRouter(prefix="/api", tags=["replay"])

TICK_SECONDS = 0.5
MAX_ALERTS_PER_TICK = 40
TAPE_ROWS = 10
ALERT_COLS = ["txn_id", "user", "card", "amount", "merchant", "city", "state", "channel", "errors", "is_fraud"]


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=float)}\n\n"


@router.get("/replay")
async def replay(request: Request, model: str, split: str | None = None,
                 t: float | None = Query(None, ge=0, le=1),
                 speed: float = Query(60, gt=0, le=10080, description="simulated minutes per real second"),
                 start: str | None = Query(None, description="simulated start, e.g. 2018-03-01")):
    d = store.get(split)
    if model not in d.models:
        raise HTTPException(404, f"unknown model '{model}'")
    t = d.models_meta[model]["threshold"] if t is None else t
    ts, scores, y = d.ts_sec, d.models[model].scores, d.y
    flagged = scores >= t
    drift = d.meta["drift"]["daily"]

    sim = int(ts[0]) if start is None else int(pd.Timestamp(start).timestamp())
    sim = max(sim, int(ts[0]))
    step = int(speed * 60 * TICK_SECONDS)

    def day_summary(day_start: int) -> dict:
        lo, hi = np.searchsorted(ts, [day_start, day_start + 86400])
        f, yy = flagged[lo:hi], y[lo:hi]
        tp = int((f & (yy == 1)).sum())
        date = str(np.datetime64(day_start, "s").astype("datetime64[D]"))
        return {"date": date, "transactions": int(hi - lo), "alerts": int(f.sum()), "true_alerts": tp,
                "false_alerts": int(f.sum()) - tp, "frauds": int(yy.sum()), "missed": int(yy.sum()) - tp,
                "fraud_amount": float(np.abs(d.amount[lo:hi][yy == 1]).sum()),
                "psi": drift.get(date, {})}

    async def stream():
        nonlocal sim
        cursor = int(np.searchsorted(ts, sim))
        day = sim - sim % 86400
        yield _sse("start", {"model": model, "threshold": t, "split": d.split, "sim_time": sim,
                             "speed": speed, "end_time": int(ts[-1])})
        while cursor < len(ts):
            if await request.is_disconnected():
                return
            nxt = sim + step
            hi = int(np.searchsorted(ts, nxt))
            idx = np.arange(cursor, hi)
            alert_idx = idx[flagged[idx]]
            alerts = d.tx.iloc[alert_idx[:MAX_ALERTS_PER_TICK]][ALERT_COLS].copy()
            alerts["ts"] = ts[alert_idx[:MAX_ALERTS_PER_TICK]]
            alerts["score"] = scores[alert_idx[:MAX_ALERTS_PER_TICK]]
            # an even sample of all traffic for the ticker tape (alerts are always included)
            tape_idx = idx[np.linspace(0, len(idx) - 1, min(TAPE_ROWS, len(idx))).astype(int)] if len(idx) else idx
            tape_idx = np.union1d(tape_idx, alert_idx[:MAX_ALERTS_PER_TICK])
            tape = d.tx.iloc[tape_idx][["txn_id", "user", "card", "amount", "city", "state", "channel"]].copy()
            tape["ts"] = ts[tape_idx]
            tape["score"] = scores[tape_idx]
            yield _sse("tick", {"sim_time": nxt, "transactions": int(len(idx)),
                                "tape_rows": tape.to_dict(orient="records"),
                                "alerts": int(len(alert_idx)),
                                "true_alerts": int(y[alert_idx].sum()),
                                "frauds": int(y[idx].sum()),
                                "amount": float(np.abs(d.amount[idx]).sum()),
                                "alert_rows": alerts.to_dict(orient="records")})
            while nxt >= day + 86400:                    # one or more days finished
                yield _sse("day", day_summary(day))
                day += 86400
            cursor, sim = hi, nxt
            await asyncio.sleep(TICK_SECONDS)
        yield _sse("end", {"sim_time": sim})

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
