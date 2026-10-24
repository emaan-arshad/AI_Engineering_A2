"""FastAPI backend. The OpenRouter key stays here (env var) and never reaches the browser."""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .adapters import (MOCK_PROFILES, MODEL_REGISTRY, build_adapters, ollama_registry,
                       ollama_url)
from .benchmark import SUITES, Config, Run, make_report, run_benchmark

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

app = FastAPI(title="Rubik's Cube Race")
_state: dict = {"run": None, "task": None, "client": None}


class RunRequest(BaseModel):
    suite: str = "primary"
    n_puzzles: int = Field(5, ge=1, le=50)
    seed: int = 42
    max_moves: int = Field(30, ge=1, le=60)
    max_retries: int = Field(2, ge=0, le=5)
    timeout_s: float = Field(60.0, ge=1, le=900)
    models: list[str] = ["jev", "luna", "clef", "opus"]
    parallel: bool = True
    mock: bool = False


def _api_key() -> str | None:
    return os.getenv("OPENROUTER_API_KEY") or None


@app.get("/api/config")
def config():
    return {
        "student_name": os.getenv("STUDENT_NAME", ""),
        "student_roll": os.getenv("STUDENT_ROLL", ""),
        "api_key_configured": bool(_api_key()),
        "suites": {k: list(v) for k, v in SUITES.items()},
        "models": {k: {"label": v[0], "category": v[1], "model_id": v[2], "interface": v[3]}
                   for k, v in {**MODEL_REGISTRY, **ollama_registry()}.items()},
        "mock_models": {k: v[0] for k, v in MOCK_PROFILES.items()},
        "ollama_url": ollama_url(),
        "ollama_configured": bool(ollama_registry()),
    }


@app.get("/api/ollama")
async def ollama_status():
    """Is Ollama running, and which models are actually pulled on this machine?"""
    if _state["client"] is None:
        _state["client"] = httpx.AsyncClient()
    configured = {k: v[2] for k, v in ollama_registry().items()}
    try:
        r = await _state["client"].get(f"{ollama_url()}/api/tags", timeout=5)
        installed = [m.get("name") for m in (r.json().get("models") or [])]
        missing = [t for t in configured.values()
                   if t not in installed and t.split(":")[0] not in [i.split(":")[0] for i in installed]]
        return {"reachable": True, "url": ollama_url(), "installed": installed,
                "configured": configured, "missing": missing}
    except Exception as e:
        return {"reachable": False, "url": ollama_url(), "installed": [],
                "configured": configured, "missing": list(configured.values()),
                "error": f"{type(e).__name__}: {e}"}


@app.post("/api/run")
async def start(req: RunRequest):
    task = _state["task"]
    if task and not task.done():
        raise HTTPException(409, "A benchmark is already running")
    if req.suite not in SUITES:
        raise HTTPException(400, f"unknown suite {req.suite}")
    if req.mock:
        keys = list(MOCK_PROFILES)
    else:
        oll = ollama_registry()
        keys = [k for k in req.models if k in MODEL_REGISTRY or k in oll]
        if not keys:
            raise HTTPException(400, "no valid models selected")
        if any(k in MODEL_REGISTRY for k in keys) and not _api_key():
            raise HTTPException(400, "OPENROUTER_API_KEY is not set on the server. Either add it to "
                                     ".env or select only local Ollama models (see .env.example).")
    if _state["client"] is None:
        _state["client"] = httpx.AsyncClient()
    cfg = Config(suite=req.suite, n_puzzles=req.n_puzzles, seed=req.seed, max_moves=req.max_moves,
                 max_retries=req.max_retries, timeout_s=req.timeout_s, models=keys, parallel=req.parallel)
    adapters = build_adapters(keys, _state["client"], cfg.timeout_s, _api_key(), cfg.seed)
    run = Run(cfg)
    _state["run"] = run
    _state["task"] = asyncio.create_task(run_benchmark(run, adapters, ROOT / "results"))
    return {"started": True, "models": keys}


@app.post("/api/stop")
async def stop():
    task = _state["task"]
    if task and not task.done():
        task.cancel()
    return {"stopped": True}


@app.get("/api/status")
def status():
    run = _state["run"]
    return run.snapshot() if run else {"status": "idle", "trials": [], "puzzles": [], "summary": {},
                                       "labels": {}, "current_puzzle": 0}


@app.get("/api/report", response_class=PlainTextResponse)
def report():
    run = _state["run"]
    return make_report(run) if run else "No run yet."


@app.on_event("shutdown")
async def _close():
    if _state["client"]:
        await _state["client"].aclose()


app.mount("/", StaticFiles(directory=ROOT / "frontend", html=True), name="frontend")
