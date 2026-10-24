"""Benchmark runner: identical puzzles, rules, timeout and retry policy for every model."""
from __future__ import annotations

import asyncio
import json
import random
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

from .adapters import BaseAdapter, MODEL_REGISTRY, MOCK_PROFILES
from .cube import SOLVED, apply_move, apply_moves, generate_scramble, is_solved
from .solver import optimal_distance

SUITES = {
    "warmup": (3, 4),
    "primary": (4, 5),
    "extension": (15, 20),
}
RECENT_HISTORY = 6  # same recent-move history policy for every model


@dataclass
class Config:
    suite: str = "primary"
    n_puzzles: int = 5
    seed: int = 42
    max_moves: int = 30
    max_retries: int = 2          # same retry policy for every model
    timeout_s: float = 60.0       # same timeout for every model
    models: list[str] = field(default_factory=lambda: ["jev", "luna", "clef", "opus"])
    parallel: bool = True


def make_puzzles(cfg: Config) -> list[dict]:
    lo, hi = SUITES[cfg.suite]
    rng = random.Random(cfg.seed)
    puzzles = []
    while len(puzzles) < cfg.n_puzzles:
        n = rng.randint(lo, hi)
        scr = generate_scramble(n, rng)
        state = apply_moves(SOLVED, scr)
        if is_solved(state):
            continue
        puzzles.append({
            "id": len(puzzles) + 1,
            "scramble": scr,
            "scramble_len": n,
            "state": state,
            "optimal": optimal_distance(state),  # exact when <= 6, else None
        })
    return puzzles


def new_trial(model_key: str, label: str, puzzle: dict) -> dict:
    return {
        "model": model_key, "label": label, "puzzle_id": puzzle["id"],
        "state": puzzle["state"], "status": "pending",   # pending|running|solved|limit
        "history": [], "moves": 0, "turns": 0, "invalid": 0, "retries": 0, "transport_errors": 0,
        "total_time_s": None, "optimal": puzzle["optimal"], "gap": None,
        "cost": 0.0, "current": None, "log": [],
    }


class Run:
    """Mutable state shared with the API (single asyncio loop -> no locking needed)."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.status = "idle"      # idle|running|done|stopped|error
        self.error: str | None = None
        self.puzzles: list[dict] = []
        self.trials: list[dict] = []
        self.current_puzzle = 0
        self.started_at = None
        self.finished_at = None
        self.labels: dict[str, str] = {}

    def snapshot(self) -> dict:
        return {
            "status": self.status, "error": self.error, "config": self.cfg.__dict__,
            "current_puzzle": self.current_puzzle, "puzzles": self.puzzles,
            "trials": self.trials, "labels": self.labels,
            "summary": summarize(self), "started_at": self.started_at, "finished_at": self.finished_at,
        }


async def _turn_with_policy(adapter: BaseAdapter, trial: dict, cfg: Config, recent, remaining):
    """One game turn = up to 1+max_retries attempts. Invalid outputs and transport errors are
    logged and counted - never silently converted into a move."""
    t_turn = time.perf_counter()
    last = None
    attempts = 0
    for attempt in range(cfg.max_retries + 1):
        attempts += 1
        last = await adapter.choose_move(trial["state"], recent, remaining)
        if last.valid:
            break
        if last.kind == "invalid":
            trial["invalid"] += 1
        else:
            trial["transport_errors"] += 1
        trial["log"].append({"turn": trial["turns"] + 1, "attempt": attempt + 1,
                             "kind": last.kind, "error": last.error})
        if attempt < cfg.max_retries:
            trial["retries"] += 1
    return last, attempts, time.perf_counter() - t_turn


async def play_trial(adapter: BaseAdapter, trial: dict, cfg: Config):
    t0 = time.perf_counter()
    trial["status"] = "running"
    while trial["turns"] < cfg.max_moves:
        remaining = cfg.max_moves - trial["turns"]
        recent = [h["move"] for h in trial["history"] if h["move"]][-RECENT_HISTORY:]
        res, attempts, turn_time = await _turn_with_policy(adapter, trial, cfg, recent, remaining)
        trial["turns"] += 1
        entry = {"turn": trial["turns"], "move": res.move if res.valid else None,
                 "latency_s": turn_time,            # what the player waits for this move
                 "attempt_latency_s": res.latency_s, "attempts": attempts, "valid": res.valid,
                 "confidence": res.confidence, "error": None if res.valid else res.error}
        trial["history"].append(entry)
        if res.cost:
            trial["cost"] += res.cost
        if res.valid:
            trial["state"] = apply_move(trial["state"], res.move)
            trial["moves"] += 1
            trial["current"] = res.move
        else:
            trial["current"] = "FORFEIT"    # retries exhausted: turn consumed, cube unchanged
        trial["total_time_s"] = time.perf_counter() - t0
        if is_solved(trial["state"]):
            trial["status"] = "solved"
            break
    else:
        trial["status"] = "limit"          # reaching the limit = unsuccessful
    trial["total_time_s"] = time.perf_counter() - t0
    if trial["status"] == "solved" and trial["optimal"] is not None:
        trial["gap"] = trial["moves"] - trial["optimal"]


async def run_benchmark(run: Run, adapters: dict[str, BaseAdapter], results_dir: Path | None = None):
    cfg = run.cfg
    try:
        run.status, run.started_at = "running", time.time()
        run.labels = {k: a.label for k, a in adapters.items()}
        run.puzzles = make_puzzles(cfg)
        for p in run.puzzles:
            run.current_puzzle = p["id"]
            trials = [new_trial(k, a.label, p) for k, a in adapters.items()]
            run.trials.extend(trials)
            coros = [play_trial(adapters[t["model"]], t, cfg) for t in trials]
            if cfg.parallel:
                await asyncio.gather(*coros)
            else:
                for c in coros:
                    await c
        run.status = "done"
    except asyncio.CancelledError:
        run.status = "stopped"
        raise
    except Exception as e:  # pragma: no cover
        run.status, run.error = "error", f"{type(e).__name__}: {e}"
    finally:
        run.finished_at = time.time()
        if results_dir and run.trials:
            results_dir.mkdir(exist_ok=True)
            path = results_dir / f"run_{int(run.finished_at)}.json"
            path.write_text(json.dumps(run.snapshot(), indent=2))


def _mean(xs):
    return statistics.fmean(xs) if xs else None


def _pct(xs, q):
    if not xs:
        return None
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


def summarize(run: Run) -> dict:
    out = {}
    for key in run.labels:
        ts = [t for t in run.trials if t["model"] == key and t["status"] in ("solved", "limit")]
        if not ts:
            continue
        lat = [h["latency_s"] for t in ts for h in t["history"]]
        solved = [t for t in ts if t["status"] == "solved"]
        gaps = [t["gap"] for t in solved if t["gap"] is not None]
        out[key] = {
            "label": run.labels[key],
            "trials": len(ts),
            "solved": len(solved),
            "success_rate": 100.0 * len(solved) / len(ts),
            "avg_latency_s": _mean(lat),
            "median_latency_s": _pct(lat, 0.5),
            "p95_latency_s": _pct(lat, 0.95),
            "avg_total_time_s": _mean([t["total_time_s"] for t in ts]),
            "avg_moves": _mean([t["moves"] for t in ts]),
            "avg_moves_solved": _mean([t["moves"] for t in solved]),
            "avg_gap": _mean(gaps),
            "optimal_hits": sum(1 for g in gaps if g == 0),
            "gap_samples": len(gaps),
            "invalid_outputs": sum(t["invalid"] for t in ts),
            "retries": sum(t["retries"] for t in ts),
            "transport_errors": sum(t["transport_errors"] for t in ts),
            "total_cost": sum(t["cost"] for t in ts),
        }
    return out


def _f(x, n=2, unit=""):
    return "n/a" if x is None else f"{x:.{n}f}{unit}"


def make_report(run: Run) -> str:
    s = summarize(run)
    if not s:
        return "No completed trials yet."
    L = [f"# Rubik's Cube Race - results ({run.cfg.suite} suite, {run.cfg.n_puzzles} puzzles, "
         f"seed {run.cfg.seed}, move limit {run.cfg.max_moves})", "",
         "| Model | Solve rate | Avg latency/move (s) | Avg total time (s) | Avg moves | Avg optimality gap | Optimal solves | Invalid | Retries |",
         "|---|---|---|---|---|---|---|---|---|"]
    for k, v in s.items():
        L.append(f"| {v['label']} | {v['solved']}/{v['trials']} ({v['success_rate']:.0f}%) | "
                 f"{_f(v['avg_latency_s'], 3)} | {_f(v['avg_total_time_s'], 2)} | {_f(v['avg_moves'], 1)} | "
                 f"{_f(v['avg_gap'], 2)} | {v['optimal_hits']}/{v['gap_samples']} | "
                 f"{v['invalid_outputs']} | {v['retries']} |")
    fastest = min(s.items(), key=lambda kv: kv[1]["avg_latency_s"] if kv[1]["avg_latency_s"] is not None else 1e9)
    best_rate = max(v["success_rate"] for v in s.values())
    top = [v["label"] for v in s.values() if v["success_rate"] == best_rate]
    gapped = {k: v for k, v in s.items() if v["avg_gap"] is not None}
    L += ["", "## Answers (auto-drafted from this run's numbers - review and edit)", "",
          f"1. **Fastest decisions:** {fastest[1]['label']} ({_f(fastest[1]['avg_latency_s'], 3)} s per move on average).",
          ("2. **Highest solve rate:** none - no model solved any puzzle within the move limit (all 0%)."
           if best_rate == 0 else
           f"2. **Highest solve rate:** {', '.join(top)} ({best_rate:.0f}%)." if len(top) < len(s) else
           f"2. **Highest solve rate:** all models tied at {best_rate:.0f}%.")]
    if gapped:
        bk, bv = min(gapped.items(), key=lambda kv: kv[1]["avg_gap"])
        L.append(f"3. **Shortest solutions:** {bv['label']} - average gap {bv['avg_gap']:.2f} moves over the exact "
                 f"optimum ({bv['optimal_hits']}/{bv['gap_samples']} solves were optimal). Gap = model moves - BFS shortest distance.")
        same = fastest[0] == bk
        L.append(f"4. **Fastest also best quality?** {'Yes' if same else 'No'} - fastest is {fastest[1]['label']}, "
                 f"best solution quality is {bv['label']}.")
    else:
        L.append("3. **Shortest solutions:** no solved puzzle with a known exact optimum in this run "
                 "(extension suite has no exact optimum; use the primary suite).")
        L.append("4. **Fastest also best quality?** Not determinable without optimality data.")
    L.append("5. **Which would I use?** Write this yourself from the table above: compare latency vs solve rate "
             "vs gap, and consider a hybrid (fast decision model proposes, frontier LLM verifies/escalates when "
             "confidence is low). Cite your own numbers as evidence.")
    return "\n".join(L)
