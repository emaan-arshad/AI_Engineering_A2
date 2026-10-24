# Rubik's Cube Race — Benchmark Report

**Student:** Emaan Arshad  **Roll No:** 23I2560
**Course:** AI Engineering — Assignment 02
**Raw data:** `results/run_*.json` (6 real OpenRouter runs + 1 clearly-labelled offline UI test)

---

## 1. Headline finding

**The frontier LLM was the only model that ever solved a cube — and when it did, it solved
it perfectly.** Claude Opus 5.5 solved a 4-move scramble in exactly **4 moves** (optimality
gap **0**). The three System-1 decision models solved **0 of 26** trials between them,
including trials where they were given a **60-move budget** for a puzzle solvable in 4.

The speed ordering is the exact reverse. The decision models answered in **0.77–0.97 s per
move**; Opus took **16.3 s per move**, roughly **20× slower** and about **190× more
expensive per move**.

So the trade-off the assignment asks about is real, and it is stark: on this task speed and
solution quality were in direct opposition, and **nothing in the middle existed**. The fast
models were not "worse but usable" — they made literally zero progress, for a reason that is
mathematically provable (§5).

---

## 2. What was completed, and what was not

All numbers below come from **real recorded API calls** to the four specified models through
OpenRouter. The raw JSON for every run is in `results/`.

**Completed:** a deterministic cube engine and exact BFS solver (both verified by a 25-test
suite); six benchmark runs against all four models; per-move latency, total trial time, move
counts, solve rate, optimality gap, invalid-output counts, retry counts, transport-error
counts and per-model cost.

**Not completed:** no single run finished all 5 puzzles with all 4 models. The class-issued
OpenRouter key carried a **$5 cap**, and $2.81 of measured spend across these runs (plus
earlier shared use) exhausted it mid-experiment.

### 2.1 Why there is no screen recording

The assignment requires a recording of the models playing and then completing the game. I
could not produce an honest one, and the raw data shows exactly why.

As the budget ran down, OpenRouter began refusing requests:

```
HTTP 402: "This request would exceed your available credits given your current
in-flight requests."   reason: in_flight_budget_exhausted
HTTP 403: "Key limit exceeded (total limit). $5.03 of $5.00 used"
```

The final run (`run_1792746442.json`) is a complete 5-puzzle attempt in which **every model
on puzzles 2–5 made 0 moves and returned 90 transport errors each** — 20 trials, all
totally blocked. A recording made at that point would have shown four frozen cards and an
error message, not a benchmark.

Following the instructor's email of 2026-10-19 I ported the application to run on free local
models via **Ollama** (implemented, tested, documented in `README.md`). The model weights are
multi-gigabyte downloads and did not finish in time.

I chose to submit a truthful partial result rather than a recording of simulated players
presented as real models. The application does contain a `MOCK` mode for offline UI testing;
it is clearly labelled as such, **no number in this report comes from it**, and the one mock
run in `results/` is included only for completeness.

**The application is complete and runnable** (`python -m pytest tests -q` → 25 passed). I can
demonstrate it live, or supply a recording once model downloads complete.

---

## 3. Experimental design

### 3.1 State/action space

- **State:** 54-character facelet string (URFDLB order) + last 6 moves + moves remaining.
- **Action:** one of 18 half-turn-metric moves — `U U' U2 R R' R2 F F' F2 D D' D2 L L' L2 B B' B2`.
- **Goal:** all six faces uniform.

### 3.2 Fairness controls

| Control | Value |
|---|---|
| Puzzles | Generated once per run from seed 42; all models start each puzzle from the same state |
| Cube engine | One engine; move permutations derived from 3-D sticker geometry, not hand-typed |
| State text | Byte-identical payload for the Decisions API and chat-completions paths (asserted by a unit test) |
| Move limit | 30 turns (60 in one exploratory run, noted per-row). Reaching it = **unsuccessful** |
| Timeout / retries | 60 s, 2 retries — identical for every model |
| Invalid output | Logged and counted, **never** converted into a move. If all attempts fail the turn is forfeited: it consumes a turn and the cube is unchanged |
| Execution | All four models play each puzzle concurrently |

### 3.3 Interfaces

- **Jev 1.13, GPT-6 Luna Decisions, Cloudflare Clef** — `POST /api/alpha/decisions`, one typed
  `choice` question `next_move` whose 18 criteria are the legal moves. Identical wording for all
  three; only `model` differs. Answer read from `answers.next_move.choice`, with `confidence`
  and `probabilities` recorded.
- **Claude Opus 5.5** — `POST /api/v1/chat/completions`, strict JSON-schema output constrained
  to `{"move": <one of 18>}`, temperature 0.

### 3.4 Establishing the true optimum

Meet-in-the-middle BFS: three layers from the solved cube, three from the scrambled state,
minimum sum over the intersection. Exact for any true distance ≤ 6, covering the Warm-up (3–4)
and Primary (4–5) suites. Deeper states return "unknown" rather than a guess.
**Optimality gap = model moves − exact shortest moves**, defined only for solved trials.

### 3.5 Which trials count

A trial measures the *model* only if it ran to completion and was not dominated by
*account-level* failures. I therefore exclude a trial if it was stopped mid-play, or if it
averaged more than one transport error per turn (i.e. the credit-exhaustion lockout).
**24 trials excluded, 28 retained.** Every exclusion is listed in §7.

---

## 4. Results

Aggregated over all 28 clean trials from the six real runs:

| Model | Interface | Trials | Solved | Solve rate | Mean latency/move | Median | Avg moves | Optimality gap | Invalid | Transport errors | Cost |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Jev 1.13** | Decisions | 9 | 0 | **0%** | **0.823 s** | **0.774 s** | 36.7 | n/a | 0 | 0 | $0.029 |
| **GPT-6 Luna Decisions** | Decisions | 9 | 0 | **0%** | 0.877 s | 0.893 s | 36.2 | n/a | 0 | 12 | $0.049 |
| **Cloudflare Clef** | Decisions | 8 | 0 | **0%** | 0.974 s | 0.941 s | 37.5 | n/a | 0 | 0 | $0.030 |
| **Claude Opus 5.5** | Chat | 2 | **1** | **50%** | 16.290 s | 15.816 s | 13.0 | **0.00** | 2 | 28 | $0.723 |

Total measured spend across all real runs: **$2.81**.

### 4.1 The one solve, verified

Opus's solved trial (`run_1792745630.json`, puzzle 1):

| | |
|---|---|
| Scramble | `U F2 R' B2` (4 turns) |
| Exact shortest solution (BFS) | **4 moves** |
| Opus's moves | `B2 R F2 U'` |
| Result | **SOLVED**, optimality gap **0** |
| Time | 28.6 s total (8.6, 7.4, 8.1, 4.6 s per move) |
| Cost | $0.067 |

`B2 R F2 U'` is precisely the inverse of the scramble, applied in reverse order — the unique
optimal solution. I re-applied this sequence to the scrambled state through the cube engine
and confirmed it reaches the solved cube.

This trial ran with a 60-move budget, but it finished at move 4, so the assignment's 30-move
cap would never have bound. The only difference is that the model was told "60 moves remaining"
rather than 30.

### 4.2 Effect of giving the models more information

Across runs the prompt was progressively enriched — first exact sticker-permutation tables for
all 18 moves, then a full preview of the resulting cube for every legal move, which reduces the
task to "pick the best of 18 displayed outcomes."

**This did not improve the decision models' solve rate at any point.** It stayed at 0%. The
only measurable effect was higher latency as the prompt grew (Jev: 0.63 s → 0.94 s per move).

Nor was the move budget the binding constraint: in `run_1792745630.json` all three decision
models were given **60 moves** on a 4-move puzzle and still finished no closer to solved.

---

## 5. Behavioural analysis — *how* each model failed

The aggregate numbers hide the most informative result. These are the actual move sequences.

### 5.1 Jev 1.13 — inverse oscillation

```
U U' U U' U U' U U' U U' U U' U U' U U' U U' U U' U U' U U' U U' U U' U U' U U' U U' U U'
U U U' U U' U U' U R R' R R' R R' R U U' U U' U U' U
```
Sixty consecutive moves of `U U'`, briefly `R R'`, then back to `U U'`. A move followed by its
own inverse returns the cube to the state before it. Jev spent an entire 60-move budget
oscillating between two states. Reported confidence: **0.06–0.51**, mostly near 0.1.

### 5.2 Cloudflare Clef — a repeating block

```
R R' R R' R' R2 F2 F' F U' U R' R U' U' U' R R' R R' R' R' R2 R R' R R' R' R' R2
R R' R R' R' R' R2 R R' R R' R' R R' R R' R R' R R' R R' R'
```
The block `R R' R R' R' R' R2` repeats verbatim three times. Confidence: **0.015–0.40**.

### 5.3 GPT-6 Luna Decisions — runs of a single move

```
L L L U L U U U U U U U U U' R R' U U U U U R' U' R' R' L' U U' U R' R' L' U U U
L' L' L' L' U U U U U U U U U U U U U U U U U U U U U
```
Twenty-four consecutive `U` moves at the end. `U` has order 4, so every fourth repetition
returns the cube to the same state. Confidence: **0.05–0.31**.

### 5.4 Claude Opus 5.5 — a real algorithm, applied blindly

In an earlier trial Opus played, from turn 10 onward:

```
R U R' U'   R U R' U'   R U R' U'   R U
```

This is the "sexy move", a genuine speedcubing trigger. The repository's own test suite proves
(`test_sexy_move_has_order_six`) that `(R U R' U')` has **order 6** — six repetitions return
the cube exactly to where it started. Opus spent 14 of 23 moves on a sequence mathematically
guaranteed to make zero net progress, at ~$0.0125 per move.

This is a qualitatively different failure. Opus produced **correct domain knowledge** — a real
algorithm a human cuber would recognise — but failed to track whether the cube was actually
improving. The decision models never produced structure at all.

### 5.5 Why the decision models cannot do this

All three failures reduce to the same thing: the models were not searching. A Decisions-API
model is a **one-shot classifier** — it maps a state to a labelled choice in a single forward
pass. Solving a cube requires lookahead over a sequence of moves, and nothing in a single-pass
classifier performs that search. Their own confidence scores (0.015–0.51, mostly ≈0.1) were
well calibrated to the fact that they were guessing.

This also explains §4.2: adding information to the prompt cannot supply search. Even shown the
exact resulting cube for all 18 moves, a model with no lookahead has no way to tell which
outcome is *closer* to solved — "closer" is a property of the remaining search depth.

### 5.6 Reliability

GPT-6 Luna Decisions was the only model to return errors during normal operation:
`HTTP 502: "OpenAI refused to answer question next_move"` — 12 across its clean trials. Under
the retry policy these consumed their turns. Jev and Clef returned **zero** errors.

**No model ever returned a malformed or illegal move.** The strict output validation never
needed to reject anything: the JSON-schema constraint on Opus and the typed `choice` on the
Decisions API both held throughout. (Opus's 2 recorded "invalid" outputs were empty responses
during the credit-exhaustion period, not reasoning failures.)

---

## 6. Answers to the required questions

### Q1. Which model is fastest in terms of making decisions?

**Jev 1.13**, at **0.823 s mean / 0.774 s median** per move across 9 trials — the fastest on
both measures. GPT-6 Luna Decisions followed (0.877 s / 0.893 s) and Cloudflare Clef was
slowest of the three (0.974 s / 0.941 s).

The three were close, and the ordering was not stable across runs: in the two earliest runs
Luna had the lowest median (0.406 s and 0.510 s). As the prompt grew, all three converged to
within ~15% of each other, which suggests latency here is dominated by prompt processing
rather than by the models themselves.

All three were **~20× faster than Claude Opus 5.5** (16.290 s/move) and roughly **190× cheaper
per move** ($0.000089 vs $0.0168 on Opus's solved trial).

### Q2. Which model achieved the highest solve rate?

**Claude Opus 5.5 — 1 of 2 clean trials (50%).** It is the only model that ever solved a cube.

The three decision models solved **0 of 26 trials combined (0%)**, including trials with a
60-move budget on a puzzle whose optimal solution is 4 moves.

The honest caveat: Opus has only **2** clean trials, because budget exhaustion blocked or cut
short the rest. A 50% rate on n=2 is weak evidence of a *rate*. It is strong evidence of
*capability* — the other models never did it even once in 26 attempts.

### Q3. Which model found the shortest solutions, and how close were they to the known optimum?

**Claude Opus 5.5, and it was exactly optimal.** Its one solve used **4 moves** against a BFS-
established optimum of **4** — an **optimality gap of 0**. The moves were `B2 R F2 U'`, the
unique shortest solution, verified by replaying them through the cube engine.

No optimality gap can be computed for the other three models, because the gap is defined only
for solved puzzles and they solved none. For reference, they averaged 36–38 moves per trial
without reaching the solved state, against an optimum of 4–5.

### Q4. Did the fastest model also achieve the best solution quality?

**No — the ranking is exactly inverted.**

| | Fastest | Best quality |
|---|---|---|
| Jev 1.13 | ✅ 0.823 s/move | ❌ 0 solves in 9 |
| Claude Opus 5.5 | ❌ 16.290 s/move (~20× slower) | ✅ only solver; gap 0 |

The fastest model had no solution quality at all to trade away, and the slowest produced a
provably optimal solution. This is the clearest result in the experiment: **on this task, the
speed/quality trade-off was total.** There was no middle option — no model was moderately fast
and moderately good.

### Q5. Would you use the fastest decision model, the frontier LLM, or a hybrid in a real application?

> *Draft — revise to reflect your own view before submitting.*

**None of them, for this particular task — and that is the most important finding.** This
problem has an exact classical solver. The BFS in this repository finds provably optimal
solutions in milliseconds for the Primary suite, with 100% reliability and zero marginal cost.
Opus matched that optimum once, 28.6 seconds and $0.067 later, with a 50% failure rate. No
sensible system would pay ~20× the latency and ~190× the cost per move for a worse version of
something `breadth_first_search()` already does. **The first question in model selection should
be whether a model is needed at all.**

For the general class of problem — multi-step planning under a step budget — the evidence
supports a **hybrid, with a specific structure**:

- **The decision models are usable as a triage filter, not as solvers.** Their 0% solve rate
  rules them out for anything requiring lookahead. But 0.8 s and $0.0001 per call, with
  confidence scores that correctly collapsed to ≈0.1 when they were guessing (§5.5), makes them
  a cheap and *honest* first pass. A model that reliably says "I don't know" for a tenth of a
  cent has real value in a routing layer.
- **The frontier LLM needs an external state-tracker.** §5.4 is the key evidence: Opus produced
  a correct speedcubing algorithm and then looped it, unable to notice it was cycling. Its
  weakness was not knowledge but *state tracking* — exactly the thing cheap deterministic code
  does perfectly. A loop-detector costing microseconds would have caught it.
- **Cost control has to be structural.** Opus consumed $0.72 across 2 trials while the three
  decision models together spent $0.11 across 26. One unbounded frontier-model agent exhausted
  a $5 budget and blocked the whole experiment (§2.1). Any production design needs hard
  per-task spend caps, not just a move limit.

So: a fast classifier to triage, a deterministic solver wherever one exists, the frontier model
reserved for genuinely open-ended cases, and cheap deterministic code wrapped around it to
track state and cap spend. The generalisable lesson from this experiment is that **the
fast-vs-slow model choice only becomes interesting once at least one option can actually do the
task** — here the decisive variable was not speed or spend but whether search was happening at
all.

---

## 7. Limitations

1. **Sample size.** The assignment recommends ≥5 primary puzzles (20 trials). Budget exhaustion
   limited this to 28 clean trials concentrated on puzzles 1–3, with **only 2 clean Opus
   trials**. Latency figures are stable (hundreds of measurements per decision model); the
   solve-rate comparison is not.
2. **Opus is under-sampled.** Its 50% solve rate rests on n=2. The claim this report makes is
   the weaker, better-supported one: Opus *can* solve these puzzles optimally and the others
   never did.
3. **Confounded conditions.** Runs differ in prompt content and move cap (30 vs 60), so they are
   configurations rather than repeats. Pooling them is justified for latency (unaffected by cap)
   but noted as a caveat for solve rate. The one solve occurred under a 60-cap, though it
   finished at move 4.
4. **Few distinct scrambles.** Most trials use puzzles 1–3 of the seed-42 Primary suite.
5. **Excluded trials.** 24 trials were excluded by the §3.5 rule: 6 stopped mid-play to conserve
   budget, and 18 destroyed by account-level credit lockout (≥87 transport errors in 30 turns).
   No trial was excluded for performing badly.
6. **Model identifiers.** `scripts/check_models.py` reports the three decision models as absent
   from OpenRouter's public `/api/v1/models` listing; they are Decisions-only models not
   expected to appear there. All three returned valid typed answers and were billed.

---

## 8. Reproducing this

```bash
python -m venv .venv && .venv\Scripts\activate     # Windows
pip install -r requirements.txt
python -m pytest tests -q                          # 25 passed
copy .env.example .env                             # add key and student details
python -m uvicorn backend.main:app --port 8000
```

Open <http://127.0.0.1:8000>, choose the Primary suite, run. Results are written to
`results/run_<timestamp>.json`. The app also runs entirely on free local models via Ollama —
see *Running on Ollama* in `README.md`.

### Verification of the measurement apparatus

Every conclusion depends on the cube engine and BFS solver being correct, so both are tested
(25 tests in `tests/`):

- Each face turn has order 4; `X` then `X'` is identity; `X2` equals `X` twice.
- Geometry checked against a physical cube (after `U`, the R face's top row appears on F).
- `(R U R' U')` has order 6 — the property used in §5.4.
- Random scrambles always yield valid states; inverting a scramble always solves it.
- BFS distances verified against known values (`R2` → 1, `R U` → 2, sexy move → 4) and never
  exceed the scramble length.
- Both API adapters are asserted to send a byte-identical state payload.
- Opus's winning sequence `B2 R F2 U'` was independently re-applied to the scrambled state
  through the engine to confirm the solve.
