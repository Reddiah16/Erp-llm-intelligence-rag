"""Phase 3 dashboard integration: the pages of dashboard/app.py own HTTP calls
only, so the contract is tested where it is observable - the API surface the
dashboard drives, driven exactly the way the page code drives it.

Streamlit cannot render inside pytest, so the source-level guarantees for
dashboard/app.py are text contracts (the pattern
tests/test_dashboard_contract.py established in Phase 1), and the
request/response sequences are exercised against the FastAPI app with the
same fakes the API tests use. Nothing here needs a live PostgreSQL, a model
download or the chart stack.
"""
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import src.ai.history as history_module
from src.ai import assistant as assistant_module
from src.ai import sql_tools as sql_tools_module
from api import main as api_main

from .ai_fakes import (FakeDatabase, FakeLlm, install_fake_embeddings,
                       make_settings)

DASHBOARD = Path(__file__).resolve().parents[1] / "dashboard" / "app.py"

PENDING_ROWS = [{"pending_orders": 225}]
ROUTE_BLOCK = ("ROUTE: sql\nQUERY: pending_purchase_order_count\n"
               "LIMIT: NONE\nREASON: pending order count")


@pytest.fixture(autouse=True)
def isolated_history():
    """Every test starts with a fresh process-wide transcript store."""
    history_module.reset_store()
    yield
    history_module.reset_store()


@pytest.fixture
def dashboard(tmp_path, monkeypatch):
    """Hermetic settings plus a TestClient, the way /chat overrides them."""
    settings = make_settings(tmp_path)
    install_fake_embeddings(monkeypatch)
    api_main.app.dependency_overrides[api_main.settings] = lambda: settings
    yield SimpleNamespace(client=TestClient(api_main.app), settings=settings)
    api_main.app.dependency_overrides.clear()


def patch_chat_stack(monkeypatch, llm: FakeLlm, connection: FakeDatabase):
    monkeypatch.setattr(assistant_module, "LlmClient", lambda _settings: llm)
    monkeypatch.setattr(assistant_module, "open_connection",
                        lambda _settings, _connection=None:
                        (connection, "", True))


# ------------------------------------------------- dashboard source contract

def test_the_dashboard_drives_every_phase_three_endpoint():
    source = DASHBOARD.read_text(encoding="utf-8")
    for path in ('"/chat"', '"/analytics"', '"/documents/upload"',
                 '"/history"', '"/health"'):
        assert path in source, f"the dashboard never calls {path}"


def test_document_upload_ui_matches_the_api_size_limit():
    source = DASHBOARD.read_text(encoding="utf-8")
    config = (DASHBOARD.parents[1] / ".streamlit" / "config.toml")
    dockerfile = DASHBOARD.parents[1] / "Dockerfile.dashboard"

    assert "2 MB (2,000,000 bytes)" in source
    assert "MAX_DOCUMENT_UPLOAD_BYTES = 2_000_000" in source
    assert "len(content) > MAX_DOCUMENT_UPLOAD_BYTES" in source
    assert 'type=("md", "txt")' in source
    assert "200MB" not in source
    assert "[server]\nmaxUploadSize = 2" in config.read_text(encoding="utf-8")
    assert "COPY .streamlit/ .streamlit/" in dockerfile.read_text(
        encoding="utf-8")
    assert "2_000_000" in (DASHBOARD.parents[1] / "api" / "main.py").read_text(
        encoding="utf-8")


def test_the_dashboard_renders_route_citations_warnings_and_checks():
    source = DASHBOARD.read_text(encoding="utf-8")
    for needle, why in (
            ("route", "the routed source (SQL/RAG/Hybrid) must be shown"),
            ("citations", "document citations with names must be shown"),
            ("warnings", "backend warnings must reach the user"),
            ("unverified_figures", "unverified figures must be flagged")):
        assert needle in source, why


def test_the_dashboard_session_id_survives_reruns_and_is_sent():
    source = DASHBOARD.read_text(encoding="utf-8")
    assert 'st.session_state["session_id"]' in source
    assert "uuid.uuid4().hex" in source, \
        "a rotated id must follow the API's 32-character session contract"
    assert '"session_id": ensure_session()' in source, \
        "every /chat call must carry the conversation id"


def test_the_dashboard_uses_the_configured_api_location():
    source = DASHBOARD.read_text(encoding="utf-8")
    assert "ERP_API_URL" in source
    assert "http://localhost:8000" in source


def test_the_navigation_offers_the_five_pages():
    source = DASHBOARD.read_text(encoding="utf-8")
    for page in ("Overview", "AI Assistant", "Business Analytics",
                 "Documents", "History"):
        assert f'"{page}"' in source


def test_the_analytics_page_renders_every_kpi_the_brief_names():
    source = DASHBOARD.read_text(encoding="utf-8")
    for needle in ("top_products_by_revenue", "top_customers_by_revenue",
                   "products_below_safety_stock", "products_needing_reorder",
                   "pending_purchase_order_count", "pending_purchase_order_value",
                   "monthly_sales_trends", "total_sales_revenue"):
        assert needle in source
    # The two revenue grains must never be presented as one figure.
    assert "Order-level total due" in source
    assert "Line-level revenue" in source


def test_the_dashboard_does_not_recompute_phase_three_kpis_locally():
    source = DASHBOARD.read_text(encoding="utf-8")
    # The Phase 1 snapshot block is the only client-side calculation and it
    # predates the API; anything recomputing the approved queries here would
    # drift from src/ai/sql_tools.py for no benefit.
    assert "sales.groupby" not in source


# ------------------------------------------------------------ /chat session

def test_chat_twice_keeps_one_history(dashboard, monkeypatch):
    client = dashboard.client
    patch_chat_stack(monkeypatch, FakeLlm(
        [ROUTE_BLOCK, "There are 225 pending purchase orders.",
         "Confirmed: 225 pending orders."], True),
        FakeDatabase([("pending_orders", PENDING_ROWS)]))
    session = "dashboard-session-01"

    first = client.post("/chat", json={
        "question": "How many pending purchase orders are there?",
        "session_id": session}).json()
    assert first["session_id"] == session
    assert first["route"] == "sql"
    assert first["citations"] == []

    second = client.post("/chat", json={
        "question": "Confirm that figure once more",
        "session_id": session}).json()
    assert second["session_id"] == session

    transcript = client.get("/history", params={"session_id": session}).json()
    assert len(transcript["turns"]) == 2
    assert transcript["turns"][0]["question"].startswith("How many pending")
    assert transcript["turns"][0]["route"] == "sql"


def test_a_chat_display_shows_warnings_and_route_when_the_model_is_down(
        dashboard, monkeypatch):
    patch_chat_stack(monkeypatch, FakeLlm([], available=False),
                     FakeDatabase([("pending_orders", PENDING_ROWS)]))
    body = dashboard.client.post("/chat", json={
        "question": "How many pending purchase orders?",
        "session_id": "display-session"}).json()
    assert body["generated"] is False
    assert any("rendered without a model" in warning
               for warning in body["warnings"])
    assert body["route"] == "sql"
    assert body["sql"]["rows"] == PENDING_ROWS


# --------------------------------------------------------------- /analytics

def test_the_analytics_page_receives_every_approved_query(
        dashboard, monkeypatch):
    from src.ai.sql_tools import QueryResult

    def synthesise(_connection, name, **_kwargs):
        tool = sql_tools_module.ALLOWLIST[name]
        return QueryResult(name=tool.name, title=tool.title, grain=tool.grain,
                           grain_note=tool.grain_note, parameters={},
                           columns=["answer"], rows=[{"answer": name}])

    def give_connection():
        yield FakeDatabase([])

    api_main.app.dependency_overrides[api_main.database] = give_connection
    monkeypatch.setattr(sql_tools_module, "execute", synthesise)
    body = dashboard.client.get("/analytics", params={"limit": 10}).json()
    assert [query["query"] for query in body["queries"]] == \
        sql_tools_module.tool_names()
    for query in body["queries"]:
        assert {"title", "grain", "grain_note", "rows",
                "row_count"} <= set(query)


def test_analytics_reports_unavailable_queries_without_dying(
        dashboard, monkeypatch):
    def failing(_conn, name, **_kwargs):
        raise RuntimeError("canned failure")

    def give_connection():
        yield FakeDatabase([])

    api_main.app.dependency_overrides[api_main.database] = give_connection
    monkeypatch.setattr(sql_tools_module, "execute", failing)
    body = dashboard.client.get("/analytics", params={
        "question": "total_sales_revenue"}).json()
    assert body["queries"] == []
    assert body["unavailable"][0]["query"] == "total_sales_revenue"


# ------------------------------------------------------ /documents upload

def test_an_upload_through_the_documents_page_is_reported_and_indexed(
        dashboard, monkeypatch):
    client, settings = dashboard.client, dashboard.settings
    stored = settings.documents_dir / "uploads" / "dashboard_upload.md"
    response = client.post("/documents/upload", files={
        "file": ("dashboard_upload.md", b"# Uploaded\n\nDashboard words.\n")})
    assert response.status_code == 200
    body = response.json()
    assert body["document"] == "dashboard_upload.md"
    assert body["ingested"] is True
    assert body["chunks"] >= 1
    assert stored.read_text(encoding="utf-8").startswith("# Uploaded")

    health = client.get("/health").json()["components"]["documents"]
    assert health["chunks"] == body["chunks"]


def test_uploading_the_same_file_twice_stays_at_one_copy(dashboard):
    client = dashboard.client
    content = b"# Repeated\n\nThe same words every time.\n"
    first = client.post("/documents/upload", files={
        "file": ("repeat.md", content)}).json()
    second = client.post("/documents/upload", files={
        "file": ("repeat.md", content)}).json()
    assert first["ingested"] and second["ingested"]
    assert second["chunks"] == first["chunks"]
    assert second["total_in_collection"] == first["total_in_collection"]


def test_the_documents_page_refuses_a_bad_file_exactly_as_the_api_does(
        dashboard):
    response = dashboard.client.post("/documents/upload", files={
        "file": ("payload.exe", b"MZ bytes")})
    assert response.status_code == 400
    assert "documents are accepted" in response.json()["detail"]


# ------------------------------------------------------------- history page

def test_the_history_page_can_browse_other_sessions(dashboard, monkeypatch):
    client = dashboard.client
    patch_chat_stack(monkeypatch, FakeLlm(
        [ROUTE_BLOCK, "225 pending orders."], True),
        FakeDatabase([("pending_orders", PENDING_ROWS)]))
    client.post("/chat", json={
        "question": "How many pending purchase orders?",
        "session_id": "old-session"})
    client.post("/chat", json={
        "question": "How many pending purchase orders?",
        "session_id": "current-1"})

    current = client.get("/history", params={
        "session_id": "current-1", "include_sessions": True}).json()
    assert len(current["turns"]) == 1
    listed = {summary["session_id"] for summary in current["sessions"]}
    assert {"old-session", "current-1"} <= listed

    other = client.get("/history", params={"session_id": "old-session"}).json()
    assert len(other["turns"]) == 1
    assert other["session_id"] == "old-session"


def test_a_session_without_turns_reads_as_empty(dashboard):
    body = dashboard.client.get("/history", params={
        "session_id": "fresh-session"}).json()
    assert body["turns"] == []
