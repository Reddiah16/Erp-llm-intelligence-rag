"""The local-model client: parsing, availability, generation, all offline.

`LlmClient` is exercised with a fake Ollama server object passed through
the `client` constructor seam, so no socket is opened and the network is
never touched.
"""
import pytest

import src.ai.llm as llm_module
from src.ai.llm import (LlmClient, LlmProtocolError, LlmUnavailable,
                        extract_json, looks_like_sql)

from .ai_fakes import FakeLlm, make_settings

CHAT_MODEL = "fake-model"


def real_client(tmp_path, server: FakeLlm) -> LlmClient:
    return LlmClient(settings=make_settings(tmp_path), client=server)


def static_server(models: dict) -> FakeLlm:
    server = FakeLlm([], available=True)
    server.list = lambda: models
    return server


# ------------------------------------------------------------------ parsing


def test_extract_json_takes_the_whole_object():
    assert extract_json('{"route": "sql", "query": "x"}') == {
        "route": "sql", "query": "x"}


def test_extract_json_reads_inside_a_fence():
    text = "Here you go:\n```json\n{\"route\": \"rag\"}\n```"
    assert extract_json(text) == {"route": "rag"}


def test_extract_json_survives_nested_braces_in_strings():
    assert extract_json('{"a": "b}c"}') == {"a": "b}c"}


def test_extract_json_ignores_prose_around_the_object():
    assert extract_json('The answer is {"route": "sql"} thank you') == {
        "route": "sql"}


def test_extract_json_returns_none_for_plain_prose():
    assert extract_json("no object here") is None
    assert extract_json("") is None
    assert extract_json(None) is None


def test_complete_json_raises_on_a_protocol_break(tmp_path):
    client = real_client(tmp_path, FakeLlm(["route: not a brace"], True))
    with pytest.raises(LlmProtocolError):
        client.complete_json("prompt")


def test_complete_json_returns_the_object(tmp_path):
    client = real_client(tmp_path, FakeLlm(["{\"a\": 1}"], True))
    assert client.complete_json("prompt") == {"a": 1}


# ------------------------------------------------------------- sql refusal


def test_looks_like_sql_catches_the_dangerous_words():
    assert looks_like_sql("SELECT count(*) FROM erp.sales_order")
    assert looks_like_sql("DROP TABLE erp.product")
    assert looks_like_sql("nothing to see; just punctuation")
    assert looks_like_sql("a comment -- in prose")


def test_looks_like_sql_allows_ordinary_words():
    assert not looks_like_sql("revenue by month")
    assert not looks_like_sql("")


# ------------------------------------------------------------- availability


def test_availability_matches_a_pulled_model(tmp_path):
    server = static_server({"models": [{"name": "fake-model"}]})
    ok, reason = real_client(tmp_path, server).availability("fake-model")
    assert ok
    assert "fake-model" in reason


def test_availability_names_the_available_models_when_one_is_missing(tmp_path):
    server = static_server({"models": [{"name": "qwen2.5:1.5b"}]})
    ok, reason = real_client(tmp_path, server).availability("llama3.2")
    assert not ok
    assert "llama3.2" in reason and "qwen2.5:1.5b" in reason


def test_a_dead_server_reports_unavailable_without_raising(tmp_path):
    server = FakeLlm([], available=True)

    def explode():
        raise ConnectionError("refused")

    server.list = explode
    ok, reason = real_client(tmp_path, server).availability("fake-model")
    assert not ok
    assert "ConnectionError" in reason


def test_availability_is_cached_inside_the_ttl(tmp_path):
    server = static_server({"models": [{"name": "fake-model"}]})
    calls = {"list": 0}
    original = server.list

    def counting():
        calls["list"] += 1
        return original()

    server.list = counting
    client = real_client(tmp_path, server)
    for _ in range(3):
        ok, _ = client.availability("fake-model")
        assert ok
    assert calls["list"] == 1, "a probe inside the TTL must not re-list models"


def test_an_expired_probe_is_refreshed(tmp_path):
    client = real_client(tmp_path, static_server({"models": [
        {"name": "fake-model"}]}))
    client._remember("fake-model", (True, "stale", 0.0))
    assert client.cached_availability("fake-model") is None
    ok, _ = client.availability("fake-model")
    assert ok


# -------------------------------------------------------------- generation


def test_complete_sends_num_predict_and_returns_stripped_text(tmp_path):
    server = FakeLlm(["  the answer  \n"], True)
    text = real_client(tmp_path, server).complete("hello")
    assert text == "the answer"
    kwargs = server.generate_kwargs[-1]
    assert kwargs["model"] == CHAT_MODEL
    assert kwargs["options"]["temperature"] == 0
    assert kwargs["options"]["num_predict"] == 400
    assert kwargs["prompt"] == "hello"


def test_complete_refuses_when_the_model_is_not_pulled(tmp_path):
    client = real_client(tmp_path, static_server({"models": []}))
    with pytest.raises(LlmUnavailable):
        client.complete("hello")


def test_complete_raises_when_the_server_is_down(tmp_path):
    server = FakeLlm([], available=True)

    def explode():
        raise ConnectionError("refused")

    server.list = explode
    with pytest.raises(LlmUnavailable):
        real_client(tmp_path, server).complete("hello")


def test_complete_wraps_a_failing_generation_as_unavailable(tmp_path):
    server = static_server({"models": [{"name": "fake-model"}]})

    def crash(**kwargs):
        raise RuntimeError("inference crashed")

    server.generate = crash
    with pytest.raises(LlmUnavailable):
        real_client(tmp_path, server).complete("hello")


def test_max_tokens_override_reaches_the_options(tmp_path):
    server = FakeLlm(["x"], True)
    real_client(tmp_path, server).complete("hello", max_tokens=120)
    assert server.generate_kwargs[-1]["options"]["num_predict"] == 120


def test_json_mode_is_requested_through_the_protocol(tmp_path):
    server = FakeLlm(["{}"], True)
    real_client(tmp_path, server).complete("hello", json_mode=True)
    assert server.generate_kwargs[-1]["format"] == "json"
