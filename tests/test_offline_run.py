import asyncio
import json

import httpx

from backend.adapters import ChatAdapter, DecisionsAdapter, build_adapters
from backend.benchmark import Config, Run, make_puzzles, make_report, run_benchmark
from backend.cube import MOVES, SOLVED, apply_move


def test_puzzles_are_reproducible_and_have_exact_optimum():
    cfg = Config(suite="primary", n_puzzles=5, seed=1)
    a, b = make_puzzles(cfg), make_puzzles(cfg)
    assert [p["state"] for p in a] == [p["state"] for p in b]
    for p in a:
        assert p["state"] != SOLVED
        assert 1 <= p["optimal"] <= p["scramble_len"] <= 5


def test_extension_suite_has_no_exact_optimum():
    ps = make_puzzles(Config(suite="extension", n_puzzles=2, seed=1))
    assert all(15 <= p["scramble_len"] <= 20 for p in ps)


def test_full_mock_run(monkeypatch):
    monkeypatch.setenv("MOCK_SPEED", "0.05")

    async def go():
        cfg = Config(suite="primary", n_puzzles=3, seed=7, models=["mock-fast", "mock-fast2", "mock-slow"])
        async with httpx.AsyncClient() as client:
            adapters = build_adapters(cfg.models, client, cfg.timeout_s, None, cfg.seed)
            run = Run(cfg)
            await run_benchmark(run, adapters, None)
        return run

    run = asyncio.run(go())
    assert run.status == "done"
    assert len(run.trials) == 9
    # every model started from the same state on each puzzle
    for pid in (1, 2, 3):
        assert len({t["history"] and run.puzzles[pid - 1]["state"] for t in run.trials if t["puzzle_id"] == pid}) == 1
    for t in run.trials:
        assert t["status"] in ("solved", "limit") and t["turns"] <= 30
        if t["status"] == "solved":
            assert t["gap"] == t["moves"] - t["optimal"] and t["gap"] >= 0
    s = run.snapshot()["summary"]
    assert set(s) == {"mock-fast", "mock-fast2", "mock-slow"}
    assert "Answers" in make_report(run)


def _fake_openrouter(content_fn, status=200):
    def handler(request: httpx.Request):
        body = json.loads(request.content)
        if request.url.path.endswith("/decisions"):
            return httpx.Response(status, json=content_fn(body, "decisions"))
        return httpx.Response(status, json=content_fn(body, "chat"))
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def test_decisions_request_shape_and_parsing():
    seen = {}

    def fn(body, kind):
        seen.update(body)
        return {"answers": {"next_move": {"type": "choice", "choice": "R'", "confidence": 0.8,
                                          "probabilities": {"R'": 0.8}}},
                "model": body["model"], "usage": {"input_tokens": 10, "output_tokens": 2, "cost": 0.001}}

    async def go():
        client = _fake_openrouter(fn)
        ad = DecisionsAdapter("jev", "Jev", "typesafe/jev-1.13", 5, client, api_key="k")
        return await ad.choose_move(SOLVED, ["R"], 30)

    r = asyncio.run(go())
    assert r.valid and r.move == "R'" and r.confidence == 0.8 and r.cost == 0.001
    q = seen["questions"]["next_move"]
    assert q["type"] == "choice" and set(q["criteria"]) == set(MOVES)
    assert seen["state"]["cube_facelets"] == SOLVED and seen["state"]["recent_moves"] == ["R"]


def test_decisions_illegal_choice_is_counted_not_applied():
    def fn(body, kind):
        return {"answers": {"next_move": {"type": "choice", "choice": "X"}}, "model": "m",
                "usage": {"input_tokens": 1, "output_tokens": 1}}

    async def go():
        ad = DecisionsAdapter("jev", "Jev", "m", 5, _fake_openrouter(fn), api_key="k")
        return await ad.choose_move(SOLVED, [], 30)

    r = asyncio.run(go())
    assert not r.valid and r.kind == "invalid" and r.move is None


def test_chat_strict_schema_and_parsing_and_malformed():
    seen = {}

    def good(body, kind):
        seen.update(body)
        return {"choices": [{"message": {"content": '{"move": "U2"}'}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 3}}

    def bad(body, kind):
        return {"choices": [{"message": {"content": "I think R is best"}}], "usage": {}}

    async def go(fn):
        ad = ChatAdapter("opus", "Opus", "anthropic/claude-opus-5.5", 5, _fake_openrouter(fn), api_key="k")
        return await ad.choose_move(SOLVED, [], 30)

    r = asyncio.run(go(good))
    assert r.valid and r.move == "U2"
    rf = seen["response_format"]["json_schema"]
    assert rf["strict"] is True and rf["schema"]["properties"]["move"]["enum"] == MOVES
    r2 = asyncio.run(go(bad))
    assert not r2.valid and r2.kind == "invalid"


def test_retry_policy_counts_invalid_and_forfeits_turn():
    calls = {"n": 0}

    def fn(body, kind):
        calls["n"] += 1
        return {"choices": [{"message": {"content": "garbage"}}], "usage": {}}

    async def go():
        client = _fake_openrouter(fn)
        cfg = Config(n_puzzles=1, max_moves=2, max_retries=2, models=["opus"])
        ad = ChatAdapter("opus", "Opus", "anthropic/claude-opus-5.5", 5, client, api_key="k")
        run = Run(cfg)
        await run_benchmark(run, {"opus": ad}, None)
        return run

    run = asyncio.run(go())
    t = run.trials[0]
    assert t["status"] == "limit" and t["moves"] == 0 and t["turns"] == 2
    assert t["invalid"] == 6 and t["retries"] == 4 and calls["n"] == 6  # 2 turns x (1+2) attempts


def test_both_interfaces_receive_identical_state_with_move_definitions():
    import json as _json
    from backend.adapters import state_payload
    p = state_payload(SOLVED, ["R"], 30)
    assert set(p["move_definitions"]["cycles"]) == set("URFDLB")
    assert p["worked_example"]["after"] != SOLVED

    seen = {}

    def fn(body, kind):
        seen[kind] = body["state"] if kind == "decisions" else _json.loads(body["messages"][1]["content"])["state"]
        if kind == "decisions":
            return {"answers": {"next_move": {"type": "choice", "choice": "R"}}, "usage": {"input_tokens": 1, "output_tokens": 1}}
        return {"choices": [{"message": {"content": '{"move": "R"}'}}], "usage": {}}

    async def go():
        c = _fake_openrouter(fn)
        await DecisionsAdapter("jev", "J", "m", 5, c, api_key="k").choose_move(SOLVED, ["R"], 30)
        await ChatAdapter("opus", "O", "m", 5, c, api_key="k").choose_move(SOLVED, ["R"], 30)

    asyncio.run(go())
    assert seen["decisions"] == seen["chat"] == p


def test_report_when_nobody_solves_is_honest():
    from backend.benchmark import Run, new_trial
    cfg = Config(n_puzzles=1, models=["a", "b"])
    run = Run(cfg)
    run.labels = {"a": "A", "b": "B"}
    run.puzzles = [{"id": 1, "scramble": ["R"], "scramble_len": 1, "state": SOLVED, "optimal": 1}]
    for k in "ab":
        t = new_trial(k, k.upper(), run.puzzles[0])
        t.update(status="limit", moves=30, turns=30, total_time_s=5.0,
                 history=[{"latency_s": 0.5, "move": "R", "valid": True}] * 30)
        run.trials.append(t)
    rep = make_report(run)
    assert "none - no model solved any puzzle" in rep
    assert "Highest solve rate:** A, B" not in rep


def test_outcomes_are_correct_and_reasoning_flag(monkeypatch):
    from backend.adapters import state_payload
    s = apply_move(SOLVED, "R")
    p = state_payload(s, [], 30)
    assert set(p["outcome_after_each_move"]) == set(MOVES)
    assert p["outcome_after_each_move"]["R'"] == SOLVED          # inverse move solves it
    monkeypatch.setenv("SHOW_OUTCOMES", "0")
    assert "outcome_after_each_move" not in state_payload(s, [], 30)

    ad = ChatAdapter("opus", "O", "m", 5, None, api_key="k")
    assert "reasoning" not in ad.build_request(SOLVED, [], 30)
    monkeypatch.setenv("OPUS_REASONING_EFFORT", "medium")
    b = ad.build_request(SOLVED, [], 30)
    assert b["reasoning"] == {"effort": "medium"} and b["max_tokens"] >= 8000 and "temperature" not in b
    assert b["response_format"]["json_schema"]["strict"] is True   # output format unchanged


# ---------------------------------------------------------------- Ollama (local, free)
def _fake_ollama(content, status=200, capture=None):
    def handler(request: httpx.Request):
        if capture is not None:
            capture.update(json.loads(request.content))
        return httpx.Response(status, json={"message": {"role": "assistant", "content": content},
                                            "prompt_eval_count": 700, "eval_count": 12})
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def test_ollama_registry_parsing(monkeypatch):
    from backend.adapters import ollama_registry
    monkeypatch.setenv("OLLAMA_MODELS", "clef=decision, qwen3:8b=llm ,deepseek-r1:8b,")
    reg = ollama_registry()
    assert list(reg) == ["ol_clef", "ol_qwen3_8b", "ol_deepseek_r1_8b"]
    assert reg["ol_clef"][1].startswith("System-1") and reg["ol_clef"][2] == "clef"
    assert reg["ol_qwen3_8b"][2] == "qwen3:8b"
    assert reg["ol_deepseek_r1_8b"][1].startswith("Frontier")   # no "=" -> defaults to llm
    monkeypatch.setenv("OLLAMA_MODELS", "")
    assert ollama_registry() == {}


def test_parse_move_json_is_strict():
    from backend.adapters import parse_move_json
    assert parse_move_json('{"move": "R\'"}') == "R'"
    assert parse_move_json('<think>hmm, U2 looks good</think>{"move": "U2"}') == "U2"
    assert parse_move_json('```json\n{"move": "F"}\n```') == "F"
    # free text naming a move is NOT silently converted into an action
    assert parse_move_json("I think R is best") is None
    assert parse_move_json('{"move": "X"}') is None          # illegal move
    assert parse_move_json('{"next": "R"}') is None          # wrong key
    assert parse_move_json("") is None


def test_ollama_request_shape_matches_other_adapters(monkeypatch):
    from backend.adapters import OllamaAdapter, state_payload
    monkeypatch.setenv("OLLAMA_NUM_CTX", "8192")
    seen = {}

    async def go():
        c = _fake_ollama('{"move": "R"}', capture=seen)
        return await OllamaAdapter("ol_q", "qwen3:8b", "qwen3:8b", 60, c).choose_move(SOLVED, ["U"], 30)

    r = asyncio.run(go())
    assert r.valid and r.move == "R"
    assert r.cost == 0.0 and r.output_tokens == 12          # local models are free
    assert seen["model"] == "qwen3:8b" and seen["stream"] is False
    assert seen["format"]["properties"]["move"]["enum"] == MOVES
    assert seen["options"]["num_ctx"] == 8192 and seen["options"]["temperature"] == 0
    # identical state payload to the OpenRouter adapters -> fair comparison
    assert json.loads(seen["messages"][1]["content"])["state"] == state_payload(SOLVED, ["U"], 30)


def test_ollama_malformed_and_http_error_are_counted():
    from backend.adapters import OllamaAdapter

    async def go(content, status=200):
        c = _fake_ollama(content, status)
        return await OllamaAdapter("ol_q", "q", "q", 60, c).choose_move(SOLVED, [], 30)

    bad = asyncio.run(go("sure, play R"))
    assert not bad.valid and bad.kind == "invalid" and bad.move is None
    err = asyncio.run(go('{"move": "R"}', 404))
    assert not err.valid and err.kind == "transport" and "404" in err.error


def test_build_adapters_routes_ollama_keys(monkeypatch):
    from backend.adapters import ChatAdapter, OllamaAdapter, build_adapters
    monkeypatch.setenv("OLLAMA_MODELS", "clef=decision,qwen3:8b=llm")

    async def go():
        async with httpx.AsyncClient() as c:
            return build_adapters(["ol_clef", "ol_qwen3_8b", "opus"], c, 60, "key")

    ad = asyncio.run(go())
    assert isinstance(ad["ol_clef"], OllamaAdapter) and ad["ol_clef"].model_id == "clef"
    assert isinstance(ad["ol_qwen3_8b"], OllamaAdapter)
    assert isinstance(ad["opus"], ChatAdapter)      # OpenRouter path still works
