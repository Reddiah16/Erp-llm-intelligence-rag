"""Session history: bounded transcripts in process memory, nothing on disk.

The store is a plain object, so every property here is tested directly:
commit order, per-session indexes, eviction, citation reduction, and the
sessions() summary that must not leak question text.
"""
import pytest

import src.ai.history as history_module
from src.ai.history import Exchange, HistoryStore, new_session_id, \
    normalise_session_id, reset_store

CITATION = {"id": "D1", "document": "inventory_policy.md",
            "section": "2. Safety stock",
            "reference": "inventory_policy.md#2-safety-stock",
            "relevance": 0.75, "excerpt": "The buffer between demand..."}


def turn(store: HistoryStore, session: str, question: str = "q",
         answer: str = "a", route: str = "sql"):
    return store.add(session, question=question, answer=answer, route=route,
                     classifier="llm", query=None, citations=[CITATION])


# ------------------------------------------------------------------ writing


def test_a_session_records_each_question_and_answer():
    store = HistoryStore()
    turn(store, "s1", question="How many pending orders?", answer="225")
    recorded = store.transcript("s1")
    assert len(recorded) == 1
    assert recorded[0]["question"] == "How many pending orders?"
    assert recorded[0]["answer"] == "225"
    assert recorded[0]["route"] == "sql"


def test_indexes_are_per_session_and_in_order():
    store = HistoryStore()
    turn(store, "s1")
    turn(store, "s1")
    turn(store, "other")
    indexes = [turn_dict["index"] for turn_dict in store.transcript("s1")]
    assert indexes == [0, 1]
    assert store.transcript("other")[0]["index"] == 0


def test_citations_shrink_to_their_references():
    store = HistoryStore()
    turn(store, "s1")
    assert store.transcript("s1")[0]["citations"] == ["D1"]


def test_plain_string_citations_are_kept_and_empty_ones_dropped():
    store = HistoryStore()
    store.add("s1", question="q", answer="a", route="rag", citations=[
        "inventory_policy.md#2-safety-stock", "", None])
    assert store.transcript("s1")[0]["citations"] == [
        "inventory_policy.md#2-safety-stock"]


def test_question_and_answer_are_trimmed():
    store = HistoryStore()
    store.add("s1", question="  spoken question  ", answer="  \n", route="rag")
    turn_dict = store.transcript("s1")[0]
    assert turn_dict["question"] == "spoken question"
    assert turn_dict["answer"] == ""


def test_cleared_history_is_empty_and_reports_the_turn_count():
    store = HistoryStore()
    turn(store, "s1")
    turn(store, "s1")
    assert store.clear("s1") == 2
    assert store.transcript("s1") == []
    assert store.clear("nowhere") == 0


def test_turn_bound_keeps_the_latest_interactions():
    store = HistoryStore(max_turns=3)
    for number in range(5):
        turn(store, "s1", question=f"q{number}")
    questions = [turn_dict["question"] for turn_dict in store.transcript("s1")]
    assert questions == ["q2", "q3", "q4"]
    # an index is stamped at append time and never rewritten, so the least
    # recent turn inside the window can share its index with the one evicted.
    assert [turn_dict["index"] for turn_dict in store.transcript("s1")] == [2, 3, 3]


def test_session_bound_evicts_the_oldest_session():
    store = HistoryStore(max_sessions=2)
    turn(store, "old")
    turn(store, "middle")
    turn(store, "new")
    assert store.__len__() == 2
    assert "old" not in store
    assert "middle" in store and "new" in store


def test_store_bounds_must_be_positive():
    with pytest.raises(ValueError):
        HistoryStore(max_sessions=0)
    with pytest.raises(ValueError):
        HistoryStore(max_turns=0)


# ------------------------------------------------------------------- reading


def test_an_unknown_session_has_no_history():
    store = HistoryStore()
    assert store.transcript("ghost") == []
    assert store.last("ghost") is None
    assert "ghost" not in store


def test_last_returns_the_newest_exchange():
    store = HistoryStore()
    turn(store, "s1", question="first")
    turn(store, "s1", question="second")
    assert store.last("s1").question == "second"


def test_sessions_summarise_counts_without_replaying_questions():
    store = HistoryStore()
    turn(store, "alpha", question="How much revenue?", route="sql")
    turn(store, "beta", question="What is the rule?", route="rag")
    summaries = store.sessions()
    assert [summary["session_id"] for summary in summaries] == ["beta", "alpha"]
    assert summaries[0]["turns"] == 1
    assert summaries[0]["last_route"] == "rag"
    body = str(summaries)
    assert "How much revenue?" not in body
    assert "What is the rule?" not in body


def test_the_most_recently_used_session_moves_to_the_front():
    store = HistoryStore()
    turn(store, "first")
    turn(store, "second")
    turn(store, "first")
    assert [summary["session_id"] for summary in store.sessions()] == [
        "first", "second"]


# ------------------------------------------------------------- identifiers


def test_normalise_mints_an_identifier_when_none_is_given():
    for empty in (None, "", "  \n"):
        assert len(normalise_session_id(empty)) == 32


def test_new_session_ids_are_always_the_bounded_length():
    assert len(new_session_id()) == 32
    ids = {new_session_id() for _ in range(5)}
    assert len(ids) == 5


def test_a_caller_identifier_is_accepted_and_trimmed():
    assert normalise_session_id("  web-session-01 ") == "web-session-01"
    assert normalise_session_id("A_b-c123") == "A_b-c123"


def test_hostile_identifiers_are_refused():
    with pytest.raises(ValueError):
        normalise_session_id("x" * 33)
    with pytest.raises(ValueError):
        normalise_session_id("bad session id!")


# ---------------------------------------------------------------- module api


@pytest.fixture(autouse=True)
def isolated_store():
    """Every test starts with a fresh process-wide store."""
    reset_store()
    yield
    reset_store()


def test_the_default_store_is_cached_until_reset():
    first = history_module.default_store()
    assert history_module.default_store() is first
    reset_store()
    assert history_module.default_store() is not first


def test_an_exchange_round_trips_into_a_dict():
    exchange = Exchange(index=0, question="q", answer="a", route="sql",
                        classifier="llm", query=None, citations=["D1"])
    body = exchange.to_dict()
    assert body["citations"] == ["D1"]
    assert body["index"] == 0
