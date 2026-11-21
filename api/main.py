"""Fraud detection dashboard: FastAPI backend + static frontend.

    python scripts/build_dashboard_data.py --split val
    uvicorn api.main:app --port 8000        # then open http://localhost:8000
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from api.routers import performance, play, replay, transactions
from api.store import store

FRONTEND = Path(__file__).resolve().parents[1] / "dashboard"


@asynccontextmanager
async def lifespan(app: FastAPI):
    store.load()
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


@app.get("/api/health")
def health():
    return {"status": "ok", "splits": list(store.splits)}


app.mount("/", StaticFiles(directory=FRONTEND, html=True), name="dashboard")
