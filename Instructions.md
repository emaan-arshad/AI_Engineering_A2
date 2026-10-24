# Rubik's Cube Race — Fast Decision Models vs Frontier LLM

AI Engineering · Assignment 02. Benchmarks four models on the same scrambled 3×3 cube through OpenRouter:

| Model | Category | OpenRouter ID | Interface |
|---|---|---|---|
| Jev 1.13 | System-1 decision | `typesafe/jev-1.13` | `POST /api/alpha/decisions` |
| GPT-6 Luna Decisions | System-1 decision | `openai/gpt-6-luna-decisions` | `POST /api/alpha/decisions` |
| Cloudflare Clef | System-1 decision | `cloudflare/clef` | `POST /api/alpha/decisions` |
| Claude Opus 5.5 | Frontier LLM | `anthropic/claude-opus-5.5` | `POST /api/v1/chat/completions` |

The app can also run the same benchmark on **local open-source models through Ollama**, with no
API key and no cost (see "Running on Ollama" below). Both paths share one cube engine, one state
payload, one move-validation rule and one retry/timeout policy, so the comparison stays fair.

**Results and analysis: see [REPORT.md](REPORT.md)** — benchmark tables, behavioural
analysis, answers to the five assignment questions, and a statement of what was and
was not completed.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                   # then edit .env
```

Fill in `.env`:
- `OPENROUTER_API_KEY` — stays on the backend; the browser never sees it.
- `STUDENT_NAME`, `STUDENT_ROLL` — shown in a fixed banner on every screen (required for the demo recording).

Verify model IDs (assignment says IDs can change), then run the tests and the app:

```bash
python -m scripts.check_models
python -m pytest tests -q          # 17 offline tests, no API key needed
python -m uvicorn backend.main:app --port 8000
# open http://localhost:8000
```

## How it works

- **State/action space.** State = 54-character facelet string (URFDLB order) + last 6 moves + moves remaining. Action = one of 18 moves (`U U' U2 R R' R2 …`, half-turn metric).
- **Cube engine** (`backend/cube.py`). Move permutations are derived from 3D sticker geometry; tests check order-4 turns, inverses, real-cube behaviour (`U` sends R's top row to F), and that `(R U R' U')` has order 6. Solved = all six faces uniform.
- **Exact optimum** (`backend/solver.py`). Meet-in-the-middle BFS (3 layers from each side) gives the *exact* shortest distance whenever it is ≤ 6 — covers warm-up (3–4) and primary (4–5) puzzles. Extension puzzles (15–20) have no exact optimum (`n/a`).
- **Optimality gap** = model moves − exact shortest moves (solved puzzles only).
- **Decisions API** (`backend/adapters.py`). One `choice` question `next_move`; identical instructions/criteria/state for all three decision models, only `model` changes. Reads `answers.next_move.choice`, `confidence`, `probabilities`, `usage.cost`. Schema checked against the OpenRouter docs (alpha endpoint — may change).
- **Opus** via chat completions with strict JSON-schema output `{"move": <enum of 18 moves>}`, temperature 0.
- **Fairness.** Same puzzles (seeded), same engine, same state text, same timeout (60 s) and retry policy (2 retries) for every model. Invalid/malformed outputs and transport errors are counted and logged, never converted into a move. If retries are exhausted the turn is forfeited (consumes one of the 30 turns, cube unchanged). Reaching 30 turns = unsuccessful.
- **Latency** = wall-clock time per turn (including retries) — what the player actually waits. Per-attempt latency is also stored. **Total solve time** = whole trial wall-clock.
- Models run in parallel per puzzle by default (untick *Run models in parallel* for strictly sequential timing). Results are saved to `results/run_<timestamp>.json`.

## Running on Ollama (free, local)

1. Install Ollama from <https://ollama.com> and make sure it is running (`ollama serve`).
2. Pull the models your instructor named, e.g.:
   ```bash
   ollama pull clef          # System-1 decision model
   ollama pull qwen3:8b      # frontier LLM
   ollama pull deepseek-r1:8b
   ollama pull gemma3:4b
   ```
   Check what you actually have with `ollama list`, and put those exact tags in `.env`.
3. Set `OLLAMA_MODELS` in `.env` (tag=category, `decision` or `llm`). Leave `OPENROUTER_API_KEY` empty.
4. Start the app. Local models appear in the model list marked `(local)` and are ticked by default,
   and a status line tells you whether Ollama is reachable and which models still need pulling.
5. Raise **Timeout (s)** to 180 or more: a local model on CPU is much slower than a hosted one.

**Design note for the write-up.** Ollama has no Decisions API, so on this path *every* model answers
through the chat endpoint under the same JSON-schema constraint. The System-1 vs frontier distinction
is then a property of the models themselves, not of two different API surfaces. Say so in your report
if you benchmark locally. Everything else is unchanged: same puzzles, same engine, same state text,
same 30-move limit, same retry and timeout policy, malformed outputs logged and counted.

**Strict output validation.** A local reply counts as a move only if, after stripping a `<think>`
block or a markdown fence, it is JSON carrying one of the 18 legal moves. A move merely mentioned in
prose is recorded as malformed, never applied.


## Suggested run for the submission

1. **Warm-up** (3 puzzles) — confirms all four adapters work.
2. **Primary** — 10 puzzles (≥ 5 required), seed 42. This is the run for the exact-gap analysis.
3. **Extension** — 5 puzzles, observe failure rates.

## Recording checklist (all four points are mandatory)

1. In-progress: start a run and show the cards updating live (current move, history).
2. Completion: show cards ending in **SOLVED** / **MOVE LIMIT REACHED**.
3. Final slide: click **📊 Benchmark slide** (success rate, latency, total time, moves, optimal gap).
4. Name + roll number: the top banner is always visible — check it isn't red ("set STUDENT_NAME…") before recording.

## Answers to the five questions

After a run, `/api/report` (also shown on the page) drafts answers 1–4 from your actual numbers. **Question 5 (what you would deploy and why) must be your own reasoning** backed by your results — e.g. latency vs solve rate vs gap, cost per move (`usage.cost` is recorded), and whether a hybrid (fast model proposes; frontier LLM verifies or takes over when confidence is low) beats either alone.

## Notes

- `MOCK` option in the UI uses simulated offline players to test the UI without a key. Never use it for the real benchmark or the submission numbers.
- Model availability, IDs, and the alpha Decisions schema can change; re-run `scripts.check_models` and the warm-up before the real run.
