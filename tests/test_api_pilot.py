"""The pilot API surface: /chat instrumented for evaluation, /feedback, the
/evaluation endpoints and the restart-persistence contract.

Same offline discipline as test_dashboard_phase3.py: the chat stack is replaced
with the scripted fakes, the pilot store points at a temporary directory (the
applied conftest autouse fixture does that already), and nothing here needs a
database, Ollama or a model download.
"""
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import src.evaluation.pilot as pilot_module
import src.ai.history as history_module
from src.ai import assistant as assistant_module
from api import main as api_main

from .ai_fakes import FakeDatabase, FakeLlm, install_fake_embeddings, \
    make_settings

PENDING_ROWS = [{"pending_orders": 225}]
ROUTE_BLOCK = ("ROUTE: sql\nQUERY: pending_purchase_order_count\n"
               "LIMIT: NONE\nREASON: pending order count")
CATALOG_QUESTION = "How many purchase orders are pending?"


@pytest.fixture(autouse=True)
def isolated_history():
    history_module.reset_store()
    yield
    history_module.reset_store()


@pytest.fixture
def pilot_store(tmp_path, monkeypatch):
    """A deterministic store directory for this test, fresh instance."""
    import pathlib
    directory = pathlib.Path(tmp_path) / "pilot"
    monkeypatch.setenv("EVALUATION_DATA_DIR", str(directory))
    pilot_module.reset_store()
    store = pilot_module.default_store()
    yield store, directory
    pilot_module.reset_store()


@pytest.fixture
def pilot_client(tmp_path, monkeypatch):
    settings = make_settings(tmp_path)
    install_fake_embeddings(monkeypatch)
    api_main.app.dependency_overrides[api_main.settings] = lambda: settings
    yield SimpleNamespace(client=TestClient(api_main.app), settings=settings,
                          store=pilot_module.default_store())
    api_main.app.dependency_overrides.clear()


def patch_chat_stack(monkeypatch, llm: FakeLlm, connection: FakeDatabase):
    monkeypatch.setattr(assistant_module, "LlmClient", lambda _settings: llm)
    monkeypatch.setattr(assistant_module, "open_connection",
                        lambda _settings, _connection=None:
                        (connection, "", True))


# ------------------------------------------------------------------ /chat

def test_a_chat_answer_records_an_evaluated_turn(pilot_client, monkeypatch):
    client, store = pilot_client.client, pilot_client.store
    patch_chat_stack(monkeypatch, FakeLlm(
        [ROUTE_BLOCK, "There are 225 pending purchase orders."], True),
        FakeDatabase([("pending_orders", PENDING_ROWS)]))

    body = client.post("/chat", json={
        "question": "How many purchase orders are there right now?",
        "session_id": "pilot-session-01",
        "tester_id": "Tester 03"}).json()

    assert body["turn_id"], "the response must carry the evaluation turn id"
    assert body["question_id"].startswith("hash") or len(body["question_id"]) == 16
    assert isinstance(body["duration_ms"], int) and body["duration_ms"] >= 0
    assert body["api_cost_inr"] == 0.0
    assert body["tester_id"] == "Tester 03"

    recorded = store.turn(body["turn_id"])
    assert recorded["session_id"] == "pilot-session-01"
    assert recorded["route"] == "sql"
    assert recorded["sql_executed"] is True
    assert recorded["sql_success"] is True
    assert recorded["tester_id"] == "Tester 03"
    assert isinstance(recorded["duration_ms"], int)


def test_the_evaluation_store_never_persists_the_question_text(
        pilot_client, monkeypatch):
    client, store = pilot_client.client, pilot_client.store
    patch_chat_stack(monkeypatch, FakeLlm(
        [ROUTE_BLOCK, "There are 225 pending purchase orders."], True),
        FakeDatabase([("pending_orders", PENDING_ROWS)]))
    response = client.post("/chat", json={
        "question": CATALOG_QUESTION + " (check the raw text is not stored)",
        "session_id": "compliance"})

    turns_text = (store.directory / "turns.jsonl").read_text(encoding="utf-8")
    assert "check the raw text is not stored" not in turns_text, \
        "AI Act compliance: query inputs are not stored beyond the session"
    assert response.status_code == 200


def test_a_catalog_question_is_recorded_under_its_curated_id(
        pilot_client, monkeypatch):
    client, store = pilot_client.client, pilot_client.store
    patch_chat_stack(monkeypatch, FakeLlm(
        [ROUTE_BLOCK, "There are 225 pending purchase orders."], True),
        FakeDatabase([("pending_orders", PENDING_ROWS)]))
    body = client.post("/chat", json={
        "question": CATALOG_QUESTION,
        "session_id": "catalog-session"}).json()
    assert body["question_id"] == "PQ-SQL-01"
    recorded = store.turn(body["turn_id"])
    assert recorded["question_kind"] == "catalog"


def test_evaluation_recording_failure_never_breaks_the_answer(
        pilot_client, monkeypatch):
    client = pilot_client.client
    patch_chat_stack(monkeypatch, FakeLlm(
        [ROUTE_BLOCK, "There are 225 pending purchase orders."], True),
        FakeDatabase([("pending_orders", PENDING_ROWS)]))

    def broken_store():
        raise RuntimeError("disk full")

    monkeypatch.setattr(pilot_module, "default_store", broken_store)
    body = client.post("/chat", json={
        "question": "How many purchase orders are there right now?",
        "session_id": "degraded"}).json()
    assert body["answer"] == "There are 225 pending purchase orders."
    assert body["turn_id"] is None
    assert "not evaluated" in body["cost_note"]


# ------------------------------------------------------------------ /feedback

def test_feedback_grades_the_recorded_turn_and_moves_the_metrics(
        pilot_client, monkeypatch):
    client, store = pilot_client.client, pilot_client.store
    patch_chat_stack(monkeypatch, FakeLlm(
        [ROUTE_BLOCK, "There are 225 pending purchase orders."], True),
        FakeDatabase([("pending_orders", PENDING_ROWS)]))
    chat = client.post("/chat", json={
        "question": CATALOG_QUESTION, "session_id": "rated-session"}).json()

    result = client.post("/feedback", json={
        "turn_id": chat["turn_id"], "helpful": True, "correct": "yes",
        "citations_relevant": "not_applicable"}).json()
    assert result["recorded"] is True
    assert result["question_id"] == "PQ-SQL-01"

    report = client.get("/evaluation/metrics").json()
    assert report["total_questions"] == 1
    assert report["answer_accuracy"]["value"] == 100.0
    assert report["answer_accuracy"]["sample"] == 1
    assert report["task_completion"]["value"] == 100.0
    assert report["accuracy_by_route"]["sql"]["sample"] == 1
    assert report["data_grounding"]["value"] == 100.0
    assert report["cost_per_query"]["value"] == 0.0
    assert report["feedback"]["turns"] == 1


def test_criteria_can_be_graded_one_at_a_time(pilot_client, monkeypatch):
    client, store = pilot_client.client, pilot_client.store
    patch_chat_stack(monkeypatch, FakeLlm(
        [ROUTE_BLOCK, "There are 225 pending purchase orders."], True),
        FakeDatabase([("pending_orders", PENDING_ROWS)]))
    chat = client.post("/chat", json={
        "question": "Another wording of the pending question",
        "session_id": "incremental"}).json()

    first = client.post("/feedback", json={
        "turn_id": chat["turn_id"], "correct": "yes"}).json()
    assert first["feedback"]["correct"] == "yes"
    assert first["feedback"]["helpful"] is None

    second = client.post("/feedback", json={
        "turn_id": chat["turn_id"], "helpful": False,
        "comment": "the figure was right but slow"}).json()
    assert second["feedback"]["correct"] == "yes", \
        "a second submission changes only the fields it names"
    assert second["feedback"]["helpful"] is False
    assert second["feedback"]["comment"].startswith("the figure")

    report = client.get("/evaluation/metrics").json()
    assert report["task_completion"]["value"] == 0.0
    assert report["feedback"]["comments"] == 1


def test_missing_or_unknown_turns_are_refused(pilot_client):
    client = pilot_client.client
    missing = client.post("/feedback", json={
        "turn_id": "", "helpful": True})
    assert missing.status_code == 422, "turn_id is required and bounded"
    unknown = client.post("/feedback", json={
        "turn_id": "does-not-exist", "helpful": True})
    assert unknown.status_code == 404
    assert "no recorded turn" in unknown.json()["detail"]


def test_invalid_and_empty_feedback_are_refused(pilot_client, monkeypatch):
    client, store = pilot_client.client, pilot_client.store
    patch_chat_stack(monkeypatch, FakeLlm(
        [ROUTE_BLOCK, "There are 225 pending purchase orders."], True),
        FakeDatabase([("pending_orders", PENDING_ROWS)]))
    chat = client.post("/chat", json={
        "question": CATALOG_QUESTION, "session_id": "grading"}).json()

    wrong_value = client.post("/feedback", json={
        "turn_id": chat["turn_id"], "correct": "not sure"})
    assert wrong_value.status_code == 422, "bad vocabulary is schema-rejected"
    oversized = client.post("/feedback", json={
        "turn_id": chat["turn_id"], "comment": "x" * 1001})
    assert oversized.status_code == 422
    empty = client.post("/feedback", json={"turn_id": chat["turn_id"]})
    assert empty.status_code == 400
    assert "feedback is empty" in empty.json()["detail"]


# ------------------------------------------------------------- /evaluation/*

def test_metrics_endpoint_reports_an_empty_store_before_any_query(pilot_client):
    report = pilot_client.client.get("/evaluation/metrics").json()
    assert report["total_questions"] == 0
    assert report["answer_accuracy"]["value"] is None
    assert report["cost_per_query"]["value"] is None


def test_the_suggested_questions_endpoint_serves_the_catalog(pilot_client):
    body = pilot_client.client.get("/evaluation/suggested-questions").json()
    ids = {entry["id"] for entry in body["questions"]}
    assert {"PQ-SQL-01", "PQ-RAG-01", "PQ-HYB-01"} <= ids
    assert body["testers"] == [f"Tester {index:02d}"
                               for index in range(1, 11)]
    assert "anonymous" in body["note"]


def test_the_export_endpoint_serves_the_csv(pilot_client, monkeypatch):
    client, store = pilot_client.client, pilot_client.store
    patch_chat_stack(monkeypatch, FakeLlm(
        [ROUTE_BLOCK, "There are 225 pending purchase orders."], True),
        FakeDatabase([("pending_orders", PENDING_ROWS)]))
    client.post("/chat", json={
        "question": CATALOG_QUESTION, "session_id": "csv-session"})

    report_path = store.directory / pilot_module.EXPORT_FILE
    report_path.write_text("existing export must remain unchanged\n",
                           encoding="utf-8")
    response = client.get("/evaluation/export")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    lines = response.text.splitlines()
    assert lines[0].startswith("timestamp,tester_id,session_id")
    assert len(lines) == 2, "header plus the one recorded turn"
    assert "PQ-SQL-01" in lines[1]
    assert report_path.read_text(encoding="utf-8") == \
        "existing export must remain unchanged\n"


def test_route_specific_metrics_separate_sql_from_rag(pilot_client, monkeypatch):
    client, store = pilot_client.client, pilot_client.store
    patch_chat_stack(monkeypatch, FakeLlm(
        [ROUTE_BLOCK, "There are 225 pending purchase orders.",
         "ROUTE: rag\nLIMIT: NONE\nREASON: policy question",
         "The policy says stock below the safety level triggers reorder."],
        True),
        FakeDatabase([("pending_orders", PENDING_ROWS)]))
    sql_chat = client.post("/chat", json={
        "question": CATALOG_QUESTION, "session_id": "routes"}).json()
    rag_chat = client.post("/chat", json={
        "question": "What does policy say about reorder timing?",
        "session_id": "routes"}).json()
    assert rag_chat["route"] == "rag"

    client.post("/feedback", json={
        "turn_id": sql_chat["turn_id"], "correct": "yes", "helpful": True})
    client.post("/feedback", json={
        "turn_id": rag_chat["turn_id"], "correct": "no",
        "citations_relevant": "yes"})

    report = client.get("/evaluation/metrics").json()
    routes = report["accuracy_by_route"]
    assert routes["sql"] == {"value": 100.0, "sample": 1}
    assert routes["rag"] == {"value": 0.0, "sample": 1}
    assert report["retrieval_accuracy"]["value"] == 100.0, \
        "the rag citations were relevant even though the answer was wrong"


# ------------------------------------------------------------- persistence

def test_metrics_survive_a_store_reset_the_way_an_api_restart_does(
        pilot_client, monkeypatch):
    client = pilot_client.client
    directory = pilot_client.store.directory
    patch_chat_stack(monkeypatch, FakeLlm(
        [ROUTE_BLOCK, "There are 225 pending purchase orders."], True),
        FakeDatabase([("pending_orders", PENDING_ROWS)]))
    chat = client.post("/chat", json={
        "question": CATALOG_QUESTION, "session_id": "restart"}).json()
    client.post("/feedback", json={
        "turn_id": chat["turn_id"], "helpful": True, "correct": "yes"})

    # simulate a process restart: forget the cached instance, keep the files
    pilot_module.reset_store()
    fresh = pilot_module.PilotStore(directory)
    report = fresh.summary_metrics()
    assert report["total_questions"] == 1
    assert report["answer_accuracy"]["value"] == 100.0
    assert report["task_completion"]["value"] == 100.0
