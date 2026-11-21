"""Fraud detection dashboard: FastAPI backend + static frontend.

    python scripts/build_dashboard_data.py --split val
    python -m api                           # then open http://localhost:8000

Configuration comes from environment variables, see api/config.py.
"""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from api.config import settings
from api.routers import performance, play, replay, transactions
from api.store import store

FRONTEND = Path(__file__).resolve().parents[1] / "dashboard"

logging.basicConfig(level=settings.log_level.upper(),
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("fraud_ops")
STARTED = {"at": None, "load_seconds": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    t0 = time.perf_counter()
    store.load()
    STARTED["at"] = time.time()
    STARTED["load_seconds"] = round(time.perf_counter() - t0, 2)
    log.info("ready: splits=%s, data loaded in %.1fs", list(store.splits), STARTED["load_seconds"])
    yield


app = FastAPI(title="Credit card fraud dashboard", version="0.1.0", lifespan=lifespan)
app.include_router(performance.router)
app.include_router(transactions.router)
app.include_router(replay.router)
app.include_router(play.router)


@app.middleware("http")
async def revalidate_frontend(request, call_next):
    """Frontend modules are edited in place with no build hashes, so always revalidate them."""
    response = await call_next(request)
    if not request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-cache"
    return response


@app.get("/health")
@app.get("/api/health")
def health():
    """Liveness/readiness: only answers once startup has loaded the data (the
    lifespan hook runs before the server accepts requests)."""
    return {"status": "ok", "splits": list(store.splits),
            "uptime_seconds": round(time.time() - STARTED["at"], 1) if STARTED["at"] else None,
            "data_load_seconds": STARTED["load_seconds"]}


app.mount("/", StaticFiles(directory=FRONTEND, html=True), name="dashboard")
