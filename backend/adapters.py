"""OpenRouter adapters. Every adapter exposes the same contract:

    await adapter.choose_move(state, recent_moves, moves_remaining) -> AttemptResult

so the benchmark runner treats all models identically. One call = one attempt;
the shared retry/timeout policy lives in benchmark.py, not here.
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from .cube import (EXAMPLE, INDEX_MAP, LAYOUT_NOTE, MOVE_CYCLES, MOVE_DESCRIPTIONS, MOVES,
                   apply_move)

DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"
CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"

QUESTION_INSTRUCTIONS = (
    "Choose the single next move most likely to progress toward the solved cube."
)

# Registry: id shown in the UI -> (label, category, model id, interface)
MODEL_REGISTRY = {
    "jev": ("Jev 1.13", "System-1 decision model", "typesafe/jev-1.13", "decisions"),
    "luna": ("GPT-6 Luna Decisions", "System-1 decision model", "openai/gpt-6-luna-decisions", "decisions"),
    "clef": ("Cloudflare Clef", "System-1 decision model", "cloudflare/clef", "decisions"),
    "opus": ("Claude Opus 5.5", "Frontier LLM", "anthropic/claude-opus-5.5", "chat"),
}


@dataclass
class AttemptResult:
    move: str | None = None          # validated move or None
    latency_s: float = 0.0           # wall time of this single HTTP attempt
    valid: bool = False
    error: str | None = None         # transport/HTTP/parse/validation error text
    kind: str = "ok"                 # ok | invalid | transport
    raw: Any = None
    confidence: float | None = None
    probabilities: dict | None = None
    cost: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None


def state_payload(state: str, recent_moves: list[str], moves_remaining: int) -> dict:
    """Identical state description for EVERY model (fairness)."""
    extra = {}
    if os.getenv("SHOW_OUTCOMES", "1") != "0":
        # Transition model of the environment: what the cube looks like after each legal action.
        # Same for every model. The model still has to judge which outcome is closest to solved.
        extra["outcome_after_each_move"] = {m: apply_move(state, m) for m in MOVES}
    return {
        **extra,
        "cube_facelets": state,
        "recent_moves": recent_moves,
        "moves_remaining": moves_remaining,
        "layout": LAYOUT_NOTE,
        "index_map": INDEX_MAP,
        "move_definitions": {
            "how_to_read": "Quarter turn X (clockwise): for each cycle [a,b,c,d] the sticker at index a "
                           "moves to index b, b to c, c to d, d to a. X' does the reverse direction; "
                           "X2 applies X twice. Moves only rearrange stickers; the 6 centres never move.",
            "cycles": MOVE_CYCLES,
        },
        "worked_example": EXAMPLE,
        "goal": "Reach the solved cube in as few moves as possible. Do not undo your previous move.",
    }


class BaseAdapter:
    def __init__(self, key: str, label: str, model_id: str, timeout_s: float, client: httpx.AsyncClient):
        self.key, self.label, self.model_id = key, label, model_id
        self.timeout_s = timeout_s
        self.client = client

    async def choose_move(self, state, recent_moves, moves_remaining) -> AttemptResult:
        raise NotImplementedError

    @staticmethod
    def _headers(api_key: str) -> dict:
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": os.getenv("APP_URL", "http://localhost:8000"),
            "X-Title": "Rubik's Cube Race (Assignment 02)",
        }


class DecisionsAdapter(BaseAdapter):
    """POST /api/alpha/decisions with one `choice` question (same wording for all models)."""

    def __init__(self, *a, api_key: str, **kw):
        super().__init__(*a, **kw)
        self.api_key = api_key

    def build_request(self, state, recent_moves, moves_remaining) -> dict:
        return {
            "model": self.model_id,
            "state": state_payload(state, recent_moves, moves_remaining),
            "questions": {
                "next_move": {
                    "type": "choice",
                    "instructions": QUESTION_INSTRUCTIONS,
                    "criteria": dict(MOVE_DESCRIPTIONS),
                }
            },
        }

    async def choose_move(self, state, recent_moves, moves_remaining) -> AttemptResult:
        body = self.build_request(state, recent_moves, moves_remaining)
        t0 = time.perf_counter()
        res = AttemptResult()
        try:
            r = await self.client.post(DECISIONS_URL, json=body, headers=self._headers(self.api_key),
                                       timeout=self.timeout_s)
            res.latency_s = time.perf_counter() - t0
            if r.status_code != 200:
                res.kind, res.error = "transport", f"HTTP {r.status_code}: {r.text[:300]}"
                return res
            data = r.json()
            res.raw = data
            ans = (data.get("answers") or {}).get("next_move") or {}
            usage = data.get("usage") or {}
            res.cost = usage.get("cost")
            res.input_tokens, res.output_tokens = usage.get("input_tokens"), usage.get("output_tokens")
            res.confidence = ans.get("confidence")
            res.probabilities = ans.get("probabilities")
            choice = ans.get("choice")
            if choice in MOVES:
                res.move, res.valid = choice, True
            else:
                res.kind, res.error = "invalid", f"choice not a legal move: {choice!r}"
        except (httpx.TimeoutException, asyncio.TimeoutError):
            res.latency_s = time.perf_counter() - t0
            res.kind, res.error = "transport", "timeout"
        except Exception as e:  # network / JSON decode
            res.latency_s = time.perf_counter() - t0
            res.kind, res.error = "transport", f"{type(e).__name__}: {e}"
        return res


class ChatAdapter(BaseAdapter):
    """POST /api/v1/chat/completions with a strict JSON-schema constrained {"move": ...}."""

    def __init__(self, *a, api_key: str, **kw):
        super().__init__(*a, **kw)
        self.api_key = api_key

    SYSTEM = (
        "You are playing a 3x3 Rubik's Cube one move at a time. You receive the cube as a "
        "54-character facelet string and must pick exactly ONE next move from the allowed list. "
        "Moves use standard notation: X = 90 deg clockwise, X' = 90 deg counterclockwise, X2 = 180 deg, "
        "for faces U R F D L B (clockwise as seen looking at that face). "
        'Reply with only JSON: {"move": "<one allowed move>"}.'
    )

    def build_request(self, state, recent_moves, moves_remaining) -> dict:
        user = {
            "state": state_payload(state, recent_moves, moves_remaining),
            "instructions": QUESTION_INSTRUCTIONS,
            "allowed_moves": dict(MOVE_DESCRIPTIONS),
        }
        effort = os.getenv("OPUS_REASONING_EFFORT", "").strip().lower()
        body = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": self.SYSTEM},
                {"role": "user", "content": json.dumps(user)},
            ],
            "temperature": 0,
            "max_tokens": int(os.getenv("OPUS_MAX_TOKENS", "2048")),
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "cube_move",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {"move": {"type": "string", "enum": MOVES}},
                        "required": ["move"],
                        "additionalProperties": False,
                    },
                },
            },
        }
        if effort in ("low", "medium", "high"):
            # Optional: let the frontier LLM think before answering. Output format stays {"move": ...}.
            body["reasoning"] = {"effort": effort}
            body["max_tokens"] = max(body["max_tokens"], int(os.getenv("OPUS_REASONING_MAX_TOKENS", "8000")))
            body.pop("temperature", None)  # some providers reject temperature with thinking on
        return body

    async def choose_move(self, state, recent_moves, moves_remaining) -> AttemptResult:
        body = self.build_request(state, recent_moves, moves_remaining)
        t0 = time.perf_counter()
        res = AttemptResult()
        try:
            r = await self.client.post(CHAT_URL, json=body, headers=self._headers(self.api_key),
                                       timeout=self.timeout_s)
            res.latency_s = time.perf_counter() - t0
            if r.status_code != 200:
                res.kind, res.error = "transport", f"HTTP {r.status_code}: {r.text[:300]}"
                return res
            data = r.json()
            res.raw = data
            usage = data.get("usage") or {}
            res.cost = usage.get("cost")
            res.input_tokens, res.output_tokens = usage.get("prompt_tokens"), usage.get("completion_tokens")
            content = ((data.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
            try:
                parsed = json.loads(content)
                mv = parsed.get("move") if isinstance(parsed, dict) else None
            except json.JSONDecodeError:
                mv = None
            if mv in MOVES:
                res.move, res.valid = mv, True
            else:
                res.kind, res.error = "invalid", f"malformed/illegal output: {content[:120]!r}"
        except (httpx.TimeoutException, asyncio.TimeoutError):
            res.latency_s = time.perf_counter() - t0
            res.kind, res.error = "transport", "timeout"
        except Exception as e:
            res.latency_s = time.perf_counter() - t0
            res.kind, res.error = "transport", f"{type(e).__name__}: {e}"
        return res


# ---------------------------------------------------------------------------
# Offline MOCK adapters - for UI development / the unit tests only.
# They are clearly labelled "MOCK" everywhere and are NOT part of the real benchmark.
# ---------------------------------------------------------------------------
MOCK_PROFILES = {
    "mock-fast": ("MOCK fast model", "Mock (offline)", 0.02, 0.55),
    "mock-fast2": ("MOCK fast model 2", "Mock (offline)", 0.03, 0.35),
    "mock-slow": ("MOCK slow frontier", "Mock (offline)", 0.40, 0.95),
}


class MockAdapter(BaseAdapter):
    def __init__(self, key: str, seed: int = 0):
        label, _cat, self.base_latency, self.skill = MOCK_PROFILES[key]
        self.key, self.label, self.model_id = key, label, f"mock/{key}"
        self.rng = random.Random(f"{key}-{seed}")
        self.malformed_rate = 0.03

    async def choose_move(self, state, recent_moves, moves_remaining) -> AttemptResult:
        from .cube import apply_move
        from .solver import optimal_distance

        scale = float(os.getenv("MOCK_SPEED", "1"))
        t0 = time.perf_counter()
        await asyncio.sleep(self.base_latency * scale * self.rng.uniform(0.7, 1.4))
        res = AttemptResult(latency_s=time.perf_counter() - t0)
        if self.rng.random() < self.malformed_rate:
            res.kind, res.error = "invalid", "mock malformed output"
            return res
        move = self.rng.choice(MOVES)
        if self.rng.random() < self.skill:
            scored = [(optimal_distance(apply_move(state, m)), m) for m in MOVES]
            known = [(d, m) for d, m in scored if d is not None]
            if known:
                move = min(known)[1]
        res.move, res.valid, res.confidence = move, True, round(self.rng.uniform(0.3, 0.95), 2)
        return res


# ---------------------------------------------------------------------------
# Ollama adapters - open-source models running locally on this machine.
# No API key, no cost. Ollama exposes only a chat endpoint (there is no Decisions
# API), so every local model answers through chat with the same JSON-schema
# constraint and the SAME state payload the OpenRouter models receive.
# ---------------------------------------------------------------------------
def ollama_url() -> str:
    return os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")


def ollama_registry() -> dict:
    """Parse OLLAMA_MODELS from .env.

    Format: comma-separated `tag=category`, category being `decision` or `llm`.
    Example: OLLAMA_MODELS=clef=decision,qwen3:8b=llm,deepseek-r1:8b=llm,gemma3:4b=llm
    Returns {ui_key: (label, category, model_tag, "ollama")}.
    """
    out: dict[str, tuple[str, str, str, str]] = {}
    for item in os.getenv("OLLAMA_MODELS", "").split(","):
        item = item.strip()
        if not item:
            continue
        tag, sep, cat = item.rpartition("=")
        if not sep:                       # no "=" -> whole item is the tag
            tag, cat = item, "llm"
        tag, cat = tag.strip(), cat.strip().lower()
        if not tag:
            continue
        category = ("System-1 decision model (local)" if cat.startswith("dec")
                    else "Frontier LLM (local)")
        key = "ol_" + re.sub(r"[^A-Za-z0-9]+", "_", tag).strip("_").lower()
        out[key] = (tag, category, tag, "ollama")
    return out


_THINK_RE = re.compile(r"<think>.*?</think>", re.S)


def parse_move_json(text: str) -> str | None:
    """Strict parse of a local model's reply.

    A <think> block or a markdown fence is wrapper noise and is stripped. After that the
    reply MUST be JSON carrying a legal move. A move merely mentioned in free text is NOT
    accepted - that keeps malformed answers counted instead of silently converted into actions.
    """
    t = _THINK_RE.sub("", text or "").strip()
    if t.startswith("```"):
        parts = t.split("```")
        if len(parts) > 1:
            t = parts[1]
        t = t.strip()
        if t.lower().startswith("json"):
            t = t[4:].strip()
    try:
        obj = json.loads(t)
    except json.JSONDecodeError:
        return None
    if isinstance(obj, dict) and obj.get("move") in MOVES:
        return obj["move"]
    return None


class OllamaAdapter(BaseAdapter):
    """Local open-source model served by Ollama (POST /api/chat)."""

    SYSTEM = ChatAdapter.SYSTEM

    def build_request(self, state, recent_moves, moves_remaining) -> dict:
        user = {
            "state": state_payload(state, recent_moves, moves_remaining),
            "instructions": QUESTION_INSTRUCTIONS,
            "allowed_moves": dict(MOVE_DESCRIPTIONS),
        }
        body = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": self.SYSTEM},
                {"role": "user", "content": json.dumps(user)},
            ],
            "stream": False,
            "format": {                       # Ollama structured output: JSON schema
                "type": "object",
                "properties": {"move": {"type": "string", "enum": MOVES}},
                "required": ["move"],
            },
            "options": {
                "temperature": 0,
                # explicit context window: the default is small on some installs and would
                # silently truncate the cube state
                "num_ctx": int(os.getenv("OLLAMA_NUM_CTX", "8192")),
                "num_predict": int(os.getenv("OLLAMA_NUM_PREDICT", "512")),
            },
        }
        think = os.getenv("OLLAMA_THINK", "").strip().lower()
        if think in ("0", "false", "off"):
            body["think"] = False
        elif think in ("1", "true", "on"):
            body["think"] = True
        return body

    async def choose_move(self, state, recent_moves, moves_remaining) -> AttemptResult:
        body = self.build_request(state, recent_moves, moves_remaining)
        t0 = time.perf_counter()
        res = AttemptResult()
        try:
            r = await self.client.post(f"{ollama_url()}/api/chat", json=body, timeout=self.timeout_s)
            res.latency_s = time.perf_counter() - t0
            if r.status_code != 200:
                res.kind, res.error = "transport", f"HTTP {r.status_code}: {r.text[:300]}"
                return res
            data = r.json()
            res.raw = data
            res.cost = 0.0                      # local model: free
            res.input_tokens = data.get("prompt_eval_count")
            res.output_tokens = data.get("eval_count")
            content = ((data.get("message") or {}).get("content")) or ""
            mv = parse_move_json(content)
            if mv:
                res.move, res.valid = mv, True
            else:
                res.kind, res.error = "invalid", f"malformed/illegal output: {content[:120]!r}"
        except (httpx.TimeoutException, asyncio.TimeoutError):
            res.latency_s = time.perf_counter() - t0
            res.kind, res.error = "transport", "timeout"
        except Exception as e:
            res.latency_s = time.perf_counter() - t0
            res.kind, res.error = "transport", f"{type(e).__name__}: {e}"
        return res


def build_adapters(model_keys: list[str], client: httpx.AsyncClient, timeout_s: float,
                   api_key: str | None, seed: int = 0) -> dict[str, BaseAdapter]:
    out: dict[str, BaseAdapter] = {}
    oll = ollama_registry()
    for k in model_keys:
        if k in MOCK_PROFILES:
            out[k] = MockAdapter(k, seed)
            continue
        if k in oll:
            label, _cat, model_id, _iface = oll[k]
            out[k] = OllamaAdapter(k, label, model_id, timeout_s, client)
            continue
        label, _cat, model_id, iface = MODEL_REGISTRY[k]
        cls = DecisionsAdapter if iface == "decisions" else ChatAdapter
        out[k] = cls(k, label, model_id, timeout_s, client, api_key=api_key)
    return out
