"""The Phase 3 API surface: /chat, /analytics, /history, /documents, /health.

Every dependency that could reach the outside world is overridden through
FastAPI's dependency_overrides: the settings point at temporary
directories, the database is a FakeDatabase, the LLM a FakeLlm injected
by patching the client factory at the module boundary. The suite runs
with no Ollama, no PostgreSQL session and no model download.
"""
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import src.ai.assistant as assistant_module
import src.ai.history as history_module
from src.ai import sql_tools as sql_tools_module
from src.rag import documents as documents_module

from api import main as api_main

from .ai_fakes import FakeDatabase, FakeLlm, make_settings, \
    install_fake_embeddings

PENDING_ROWS = [{"pending_orders": 225}]
PRODUCT_ROWS = [
    {"product_id": 782, "name": "Mountain-200 Black, 38",
     "revenue": "4400592.800400", "units_sold": 2977},
]
REVENUE_ROWS = [{
    "orders": 31465, "order_merchandise": "109846381.4039",
    "order_tax": "10186974.4602", "order_freight": "3183430.2518",
    "order_total_due": "123216786.1159", "order_lines": 121317,
    "line_revenue": "109846381.399888", "units_sold": 274914,
}]

ROUTE_BLOCK = "ROUTE: sql\nQUERY: pending_purchase_order_count\nLIMIT: NONE\n" \
              "REASON: pending order count"


@pytest.fixture(autouse=True)
def isolated_history():
    history_module.reset_store()
    yield
    history_module.reset_store()


@pytest.fixture
def settings(tmp_path, monkeypatch):
    return make_settings(tmp_path)


@pytest.fixture
def client(tmp_path, monkeypatch, settings):
    """A TestClient with hermetic settings; overrides stay per-test."""
    install_fake_embeddings(monkeypatch)
    api_main.app.dependency_overrides[api_main.settings] = lambda: settings
    yield TestClient(api_main.app)
    api_main.app.dependency_overrides.clear()


def fake_sql_database() -> FakeDatabase:
    return FakeDatabase([("pending_orders", PENDING_ROWS),
                         ("order_total_due", REVENUE_ROWS),
                         ("line_total", PRODUCT_ROWS)])


def install_database_override(connection) -> None:
    """Point the /analytics database dependency at a canned connection."""
    def give_connection():
        yield connection
    api_main.app.dependency_overrides[api_main.database] = give_connection


def patch_llm(monkeypatch, instance: FakeLlm, module=assistant_module) -> None:
    """Replace the client factory so /chat never probes the local server."""
    monkeypatch.setattr(module, "LlmClient", lambda _settings: instance)


def patch_database(monkeypatch, connection, error: str = "") -> None:
    monkeypatch.setattr(assistant_module, "open_connection",
                        lambda _settings, _connection=None:
                        (connection, error, True))


# ------------------------------------------------------------------- /chat


def test_chat_answers_a_sql_question_with_data_and_session(client, monkeypatch):
    patch_llm(monkeypatch, FakeLlm([ROUTE_BLOCK,
                                    "There are 225 pending purchase orders."],
                                   True))
    patch_database(monkeypatch, fake_sql_database())
    response = client.post("/chat", json={"question": "How many pending "
                                                   "purchase orders?"})
    assert response.status_code == 200
    body = response.json()
    assert body["route"] == "sql"
    assert body["generated"] is True
    assert body["data_available"] is True
    assert body["query"] == "pending_purchase_order_count"
    assert "225" in body["answer"]
    assert body["unverified_figures"] == []
    assert body["session_id"]                # minted when none was supplied


def test_chat_cleans_up_its_per_request_connection(client, monkeypatch):
    patch_llm(monkeypatch, FakeLlm([ROUTE_BLOCK,
                                    "There are 225 pending purchase orders."],
                                   True))
    connection = fake_sql_database()
    patch_database(monkeypatch, connection)
    client.post("/chat", json={"question": "How many pending purchase orders?"})
    assert connection.rolled_back and connection.closed, \
        "the per-request connection must be cleaned up"


def test_chat_honours_a_caller_session_id(client, monkeypatch):
    patch_llm(monkeypatch, FakeLlm([ROUTE_BLOCK, "225 pending orders."], True))
    patch_database(monkeypatch, fake_sql_database())
    body = client.post("/chat", json={
        "question": "How many pending purchase orders?",
        "session_id": "browser-session-31"}).json()
    assert body["session_id"] == "browser-session-31"
    assert history_module.default_store().transcript("browser-session-31")


def test_chat_survives_an_unavailable_model_with_repeated_proof(client, monkeypatch):
    patch_llm(monkeypatch, FakeLlm([], available=False))
    patch_database(monkeypatch, fake_sql_database())
    body = client.post("/chat", json={
        "question": "How many pending purchase orders?"}).json()
    assert body["generated"] is False
    assert body["data_available"] is True
    assert any("rendered without a model" in warning
               for warning in body["warnings"])
    assert "pending_orders=225" in body["answer"]


def test_chat_reports_a_dead_database_without_failing(client, monkeypatch):
    patch_llm(monkeypatch, FakeLlm([ROUTE_BLOCK,
                                    "Business figures were unavailable."], True))
    patch_database(monkeypatch, None,
                   error="could not connect to PostgreSQL at localhost:5432")
    body = client.post("/chat", json={
        "question": "How many pending purchase orders?"}).json()
    assert body["data_available"] is False
    assert any("structured data unavailable" in warning
               for warning in body["warnings"])


def test_an_empty_question_is_a_client_error(client):
    response = client.post("/chat", json={"question": "   "})
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_an_oversized_session_id_is_rejected(client):
    response = client.post("/chat", json={
        "question": "How many pending purchase orders?",
        "session_id": "x" * 33})
    assert response.status_code == 422


# -------------------------------------------------------------- /analytics


def test_analytics_answers_one_approved_query_directly(client):
    install_database_override(fake_sql_database())
    response = client.get("/analytics", params={
        "question": "total_sales_revenue", "limit": 1})
    assert response.status_code == 200
    body = response.json()
    assert body["limit"] == 1
    assert body["queries"][0]["query"] == "total_sales_revenue"
    assert body["queries"][0]["rows"][0]["order_total_due"] == "123216786.1159"
    assert body["unavailable"] == []


def test_analytics_refuses_an_unapproved_name(client):
    install_database_override(fake_sql_database())
    response = client.get("/analytics", params={
        "question": "top_customers_by_moon_phase"})
    assert response.status_code == 404
    assert "not an approved query" in response.json()["detail"]


def test_analytics_bounds_both_the_limit_edges(client):
    install_database_override(fake_sql_database())
    assert client.get("/analytics", params={"limit": 0}).status_code == 422
    assert client.get("/analytics", params={"limit": 1001}).status_code == 422

    body_above_ceiling = client.get("/analytics", params={
        "question": "top_products_by_revenue", "limit": 200}).json()
    assert body_above_ceiling["limit"] == 200
    assert body_above_ceiling["queries"][0]["parameters"]["limit"] == 50, \
        "the configured ceiling must clamp what reaches SQL"


def test_analytics_runs_every_allowed_query_when_none_is_named(
        client, monkeypatch):
    from src.ai.sql_tools import QueryResult

    install_database_override(fake_sql_database())

    def synthesise(_connection, name, **_kwargs):
        tool = sql_tools_module.ALLOWLIST[name]
        return QueryResult(name=tool.name, title=tool.title, grain=tool.grain,
                           grain_note=tool.grain_note, parameters={},
                           columns=["answer"], rows=[{"answer": name}])

    monkeypatch.setattr(sql_tools_module, "execute", synthesise)
    body = client.get("/analytics").json()
    assert [query["query"] for query in body["queries"]] == \
        sql_tools_module.tool_names()
    assert body["unavailable"] == []


def test_analytics_reports_queries_that_cannot_run(client, monkeypatch):
    def failing(_conn, name, **_kwargs):
        raise RuntimeError("canned failure")

    install_database_override(fake_sql_database())
    monkeypatch.setattr(sql_tools_module, "execute", failing)
    body = client.get("/analytics", params={
        "question": "total_sales_revenue"}).json()
    assert body["queries"] == []
    assert body["unavailable"][0]["query"] == "total_sales_revenue"
    assert "RuntimeError" in body["unavailable"][0]["detail"]


def test_analytics_says_unavailable_when_the_database_is_dead(client):
    def dead():
        raise HTTPException(status_code=503, detail="database unreachable")

    api_main.app.dependency_overrides[api_main.database] = dead
    response = client.get("/analytics")
    assert response.status_code == 503
    assert "database unreachable" in response.json()["detail"]


def test_the_real_database_dependency_rolls_back_and_closes(
        client, monkeypatch, settings):
    """The real `database` dependency: rollback plus close, never a commit."""
    connection = fake_sql_database()
    monkeypatch.setattr(assistant_module, "open_connection",
                        lambda _settings, _connection=None: (connection, "", True))
    api_main.app.dependency_overrides.clear()
    api_main.app.dependency_overrides[api_main.settings] = lambda: settings

    response = TestClient(api_main.app).get(
        "/analytics", params={"question": "total_sales_revenue"})
    assert response.status_code == 200
    assert response.json()["queries"][0]["query"] == "total_sales_revenue"
    assert connection.rolled_back and connection.closed


# --------------------------------------------------------------- /history


def test_history_stores_a_transcript_for_a_client_session(client):
    store = history_module.default_store()
    store.add("sess-doc-1", question="How much revenue?", answer="123,216,786",
              route="sql", classifier="llm", query="total_sales_revenue",
              citations=[{"id": "S1"}])
    body = client.get("/history", params={"session_id": "sess-doc-1"}).json()
    assert len(body["turns"]) == 1
    assert body["turns"][0]["question"] == "How much revenue?"
    assert body["turns"][0]["citations"] == ["S1"]


def test_history_of_an_unknown_session_is_empty(client):
    body = client.get("/history", params={"session_id": "ghost"}).json()
    assert body["turns"] == []


def test_history_sessions_are_opt_in_and_carry_counts_only(client):
    store = history_module.default_store()
    store.add("sess-a", question="secret question", answer="a", route="rag")
    body = client.get("/history", params={"session_id": "sess-a"}).json()
    assert body["sessions"] == []

    body = client.get("/history", params={
        "session_id": "sess-a", "include_sessions": "true"}).json()
    assert len(body["sessions"]) == 1
    assert body["sessions"][0]["session_id"] == "sess-a"
    assert "secret question" not in str(body["sessions"])


def test_history_refuses_a_hostile_session_id(client):
    response = client.get("/history", params={"session_id": "x" * 40})
    assert response.status_code == 422


def test_chat_transcripts_show_up_in_history(client, monkeypatch):
    patch_llm(monkeypatch, FakeLlm([ROUTE_BLOCK, "225 pending orders."], True))
    patch_database(monkeypatch, fake_sql_database())
    body = client.post("/chat", json={"question": "How many pending "
                                                   "purchase orders?"}).json()
    transcript = client.get("/history", params={
        "session_id": body["session_id"]}).json()
    assert len(transcript["turns"]) == 1
    assert transcript["turns"][0]["question"].startswith("How many pending")


# ------------------------------------------------------- /documents/upload


def test_upload_stores_and_ingests_a_policy_document(client, settings):
    response = client.post("/documents/upload",
                           files={"file": ("uploaded.md",
                                           b"# Uploaded\n\nSome words here.")})
    assert response.status_code == 200
    body = response.json()
    assert body["document"] == "uploaded.md"
    assert body["ingested"] is True
    assert body["chunks"] >= 1
    stored = settings.documents_dir / "uploads" / "uploaded.md"
    assert stored.read_text(encoding="utf-8") == "# Uploaded\n\nSome words here."
    assert body["total_in_collection"] == body["chunks"]


def test_upload_refuses_a_traversal_name(client):
    response = client.post("/documents/upload",
                           files={"file": ("../evil.md", b"# content")})
    assert response.status_code == 400
    assert "path" in response.json()["detail"]


def test_upload_refuses_a_non_text_suffix(client):
    response = client.post("/documents/upload",
                           files={"file": ("payload.exe", b"MZ\x90\x00")})
    assert response.status_code == 400
    assert "documents are accepted" in response.json()["detail"]


def test_upload_refuses_non_utf8_bytes(client):
    response = client.post("/documents/upload",
                           files={"file": ("binary.md", b"\xff\xfe\x00")})
    assert response.status_code == 400
    assert "UTF-8" in response.json()["detail"]


def test_upload_refuses_a_payload_over_the_size_bound(client):
    response = client.post("/documents/upload",
                           files={"file": ("big.md", b"a" * (2_000_000 + 1))})
    assert response.status_code == 413


# ----------------------------------------------------------------- /health


def test_health_reports_every_phase_three_component(client, monkeypatch):
    from src.db import config as db_config
    monkeypatch.setattr(db_config, "server_reachable",
                        lambda *a, **kw: (True, "faked reachability"))
    monkeypatch.setattr(documents_module, "status", lambda *a, **kw: {
        "available": False, "chunks": 0, "documents": 0,
        "reason": "test corpus stub", "corpus_files": []})
    patch_llm(monkeypatch, FakeLlm([], True), module=api_main.llm)

    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["version"] == "1.1.0"
    components = body["components"]
    assert components["database"]["available"] is True
    assert components["llm"]["available"] is True
    assert components["documents"]["available"] is False
    assert components["documents"]["detail"] == "test corpus stub"
    assert components["queries_allowed"] == len(sql_tools_module.ALLOWLIST)


def test_health_survives_total_component_failure(client, monkeypatch):
    from src.db import config as db_config
    monkeypatch.setattr(db_config, "server_reachable",
                        lambda *a, **kw: (_ for _ in ()
                                          ).throw(RuntimeError("canned")))
    patch_llm(monkeypatch, FakeLlm([], available=False), module=api_main.llm)
    monkeypatch.setattr(documents_module, "status",
                        lambda *a, **kw: (_ for _ in ()).throw(
                            RuntimeError("canned")))

    body = client.get("/health").json()
    assert body["status"] == "ok"
    for key in ("database", "llm", "documents"):
        assert body["components"][key]["available"] is False


def test_health_surfaces_the_embedding_provider_information(client, monkeypatch):
    from src.db import config as db_config
    monkeypatch.setattr(db_config, "server_reachable",
                        lambda *a, **kw: (True, "faked reachability"))
    reported = {
        "available": True, "chunks": 9, "documents": 4, "reason": "",
        "corpus_files": ["policy.md"],
        "embedding": {"provider": "chroma", "model": "nomic-embed-text",
                      "available": False,
                      "detail": "the bundled MiniLM model is not cached yet"},
    }
    monkeypatch.setattr(documents_module, "status", lambda *a, **kw: reported)
    patch_llm(monkeypatch, FakeLlm([], True), module=api_main.llm)

    documents_component = client.get("/health").json()["components"]["documents"]
    embedding = documents_component["embedding"]
    assert embedding["available"] is False
    assert "not cached yet" in embedding["detail"], \
        "a degraded embedding provider must come with a remedy on /health"
