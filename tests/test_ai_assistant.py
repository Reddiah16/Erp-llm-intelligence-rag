"""The assistant orchestration: every combination of present and missing parts.

All tests run offline. The database is a FakeDatabase (rows served to
sql_tools exactly as psycopg would), the model is a FakeLlm, and the
document store is patched at `documents.search` because the retrieval
contract itself is asserted in tests/test_rag_documents.py.
"""
import pytest

from src.ai import routing
from src.ai.assistant import answer_question, open_connection, verify_figures
from src.db import config as db_config
from src.rag import documents

from .ai_fakes import FakeDatabase, FakeLlm, make_settings

# Canned rows: the same figures the live smoke reported, served without a
# server, so the orchestration contract can be asserted deterministically.
PENDING_ROWS = [{"pending_orders": 225}]
REVENUE_ROWS = [{
    "orders": 31465, "order_merchandise": "109846381.4039",
    "order_tax": "10186974.4602", "order_freight": "3183430.2518",
    "order_total_due": "123216786.1159", "order_lines": 121317,
    "line_revenue": "109846381.399888", "units_sold": 274914,
}]

HITS = [
    {"document": "inventory_policy.md", "document_id": "EMP-INV-001",
     "section": "2. Safety stock",
     "reference": "inventory_policy.md#2-safety-stock",
     "relevance": 0.75, "distance": 0.25,
     "text": "A product is below safety stock when on-hand quantity is "
             "strictly less than its safety stock level."},
    {"document": "business_rules.md", "document_id": "EMP-RULE-002",
     "section": "1. Two grains of money",
     "reference": "business_rules.md#1-two-grains-of-money",
     "relevance": 0.60, "distance": 0.40,
     "text": "Order-level money and line-level money are different measures "
             "and must never be added together."},
]


@pytest.fixture
def settings(tmp_path):
    return make_settings(tmp_path)


@pytest.fixture
def searchable(monkeypatch):
    """documents.search always returns the two passages, whatever the store."""
    def retrieve(*args, **kwargs):
        return [dict(hit) for hit in HITS]
    monkeypatch.setattr(documents, "search", retrieve)


def pending_database() -> FakeDatabase:
    return FakeDatabase([("pending_orders", PENDING_ROWS)])


def revenue_database() -> FakeDatabase:
    return FakeDatabase([("order_total_due", REVENUE_ROWS)])


def pending_decision() -> routing.RouteDecision:
    return routing.RouteDecision(route="sql",
                                 query="pending_purchase_order_count")


def revenue_decision() -> routing.RouteDecision:
    return routing.RouteDecision(route="sql", query="total_sales_revenue")


def rag_decision() -> routing.RouteDecision:
    return routing.RouteDecision(route="rag")


def hybrid_decision() -> routing.RouteDecision:
    return routing.RouteDecision(route="hybrid", query="total_sales_revenue")


# ------------------------------------------------------------------ sql route


def test_sql_works_while_the_model_answers_within_its_evidence(settings):
    llm = FakeLlm(["There are 225 pending purchase orders."], True)
    answer = answer_question(
        "How many pending purchase orders are there?", settings, llm,
        decision=pending_decision(), connection=pending_database())
    assert answer.generated
    assert answer.route == "sql"
    assert answer.data_available
    assert not answer.warnings
    assert answer.unverified_figures == []
    assert "225" in answer.text


def test_sql_still_answers_when_the_model_is_unreachable(settings):
    answer = answer_question(
        "How many pending purchase orders are there?", settings,
        FakeLlm([], available=False), decision=pending_decision(),
        connection=pending_database())
    assert not answer.generated
    assert answer.data_available
    assert any("rendered without a model" in warning
               for warning in answer.warnings)
    # the degraded text is evidence copied through, never a model summary
    assert "pending_orders=225" in answer.text
    assert "No language model was available" in answer.text
    assert answer.unverified_figures == []


def test_a_fabricated_figure_replaces_the_model_wording_with_the_evidence(settings):
    llm = FakeLlm(["Order-level total due is 999,999,999.99 dollars."], True)
    answer = answer_question(
        "What is the total sales revenue?", settings, llm,
        decision=revenue_decision(), connection=revenue_database())
    assert not answer.generated
    assert "999999999.99" in answer.unverified_figures
    assert any("stated figures that are not in the evidence" in warning
               for warning in answer.warnings)
    assert "order_total_due=123216786.1159" in answer.text


def test_a_correctly_rounded_copy_of_a_stored_figure_is_not_fabrication(settings):
    llm = FakeLlm(["Order-level total due is 123,216,786.12 dollars."], True)
    answer = answer_question(
        "What is the total sales revenue?", settings, llm,
        decision=revenue_decision(), connection=revenue_database())
    assert answer.generated
    assert answer.unverified_figures == []
    assert not answer.warnings


def test_an_exact_copy_at_full_precision_is_not_fabrication(settings):
    llm = FakeLlm(["Order-level total due is 123216786.1159."], True)
    answer = answer_question(
        "What is the total sales revenue?", settings, llm,
        decision=revenue_decision(), connection=revenue_database())
    assert answer.generated
    assert answer.unverified_figures == []


def test_unavailable_structured_data_is_reported_not_invented(settings, monkeypatch):
    def refuse():
        raise RuntimeError("POSTGRES_PASSWORD is not set")

    monkeypatch.setattr(db_config, "config_from_env", refuse)
    llm = FakeLlm(["The figure is unavailable right now."], True)
    answer = answer_question(
        "How many pending purchase orders are there?", settings, llm,
        decision=pending_decision(), connection=None)
    assert not answer.data_available
    assert answer.route == "sql"
    assert answer.query == "pending_purchase_order_count"
    assert any("structured data unavailable" in warning
               for warning in answer.warnings)
    assert "pending_orders" not in answer.text


# ------------------------------------------------------------------ rag route


def test_a_policy_question_returns_citations_and_generated_text(settings, searchable):
    llm = FakeLlm(["Safety stock is reviewed when demand changes. [D1]"], True)
    answer = answer_question(
        "What is the safety stock rule?", settings, llm, decision=rag_decision())
    assert answer.route == "rag"
    assert answer.generated
    assert len(answer.citations) == 2
    assert answer.citations[0]["id"] == "D1"
    assert answer.documents_available
    assert "[D1]" in llm.prompts[0], "the model must see labelled passages"
    assert answer.unverified_figures == []


def test_policy_passages_are_shown_when_the_model_is_down(settings, searchable):
    answer = answer_question(
        "What is the safety stock rule?", settings, FakeLlm([], available=False),
        decision=rag_decision())
    assert not answer.generated
    assert answer.documents_available
    assert "Policy documents retrieved:" in answer.text
    assert "[D1] inventory_policy.md" in answer.text
    assert any("rendered without a model" in warning
               for warning in answer.warnings)


def test_no_retrieval_hits_become_an_honest_unavailability(settings, monkeypatch):
    monkeypatch.setattr(documents, "search", lambda *a, **kw: [])
    llm = FakeLlm(["The policy could not be retrieved."], True)
    answer = answer_question(
        "What is the safety stock rule?", settings, llm, decision=rag_decision())
    assert not answer.documents_available
    assert any("policy documents unavailable" in warning
               for warning in answer.warnings)


def test_a_broken_retriever_is_reported_not_raised(settings, monkeypatch):
    def broken(*args, **kwargs):
        raise NotImplementedError("store offline")

    fake_status = {"reason": "the document collection has not been created"}
    monkeypatch.setattr(documents, "search", broken)
    monkeypatch.setattr(documents, "status", lambda *a, **kw: fake_status)
    llm = FakeLlm(["The policy documents are unavailable."], True)
    answer = answer_question(
        "What is the safety stock rule?", settings, llm, decision=rag_decision())
    assert not answer.documents_available
    assert any("policy documents unavailable" in warning
               for warning in answer.warnings)


# --------------------------------------------------------------- hybrid route


def test_a_hybrid_question_carries_both_figures_and_policy(settings, searchable):
    llm = FakeLlm(["Order-level total due is 123,216,786.1159. [D1]"], True)
    answer = answer_question(
        "What is the total revenue and the money rule?", settings, llm,
        decision=hybrid_decision(), connection=revenue_database())
    assert answer.route == "hybrid"
    assert answer.generated
    assert answer.data_available and answer.documents_available
    assert len(answer.citations) == 2
    assert answer.unverified_figures == []
    assert "order_total_due=123216786.1159" in llm.prompts[-1]
    assert "Answer every requested part." in llm.prompts[-1]
    assert "SQL rows returned for it" in llm.prompts[-1]


def test_hybrid_list_answer_includes_all_approved_sql_rows(settings, searchable):
    rows = [
        {"product_id": index, "name": name, "product_number": f"P-{index}",
         "reorder_point": 10, "total_quantity": 5, "shortfall": 5}
        for index, name in enumerate(("Alpha", "Bravo", "Charlie"), start=1)
    ]
    answer = answer_question(
        "Explain the procurement guidelines and identify products that need "
        "reordering.",
        settings,
        FakeLlm(["Alpha needs reordering. [D1]"], True),
        decision=routing.RouteDecision(
            route="hybrid", query="products_needing_reorder"),
        connection=FakeDatabase([("stock_position", rows)]))

    assert answer.route == "hybrid"
    assert all(name in answer.text for name in ("Alpha", "Bravo", "Charlie"))
    assert "Complete approved SQL result rows (3 returned)" in answer.text
    assert answer.unverified_figures == []


def test_a_hybrid_question_survives_both_components_being_down(settings, monkeypatch):
    monkeypatch.setattr(documents, "search", lambda *a, **kw: [])
    answer = answer_question(
        "What is the total revenue and the money rule?", settings,
        FakeLlm([], available=False), decision=hybrid_decision(),
        connection=FakeDatabase([]))
    assert not answer.generated
    assert not answer.data_available and not answer.documents_available
    assert "Business figures are unavailable" in answer.text
    assert "Policy documents are unavailable" in answer.text


# ---------------------------------------------------------- unsupported route


def test_an_unsupported_route_answers_with_the_catalogue(settings):
    answer = answer_question(
        "What is the weather on the moon?", settings, FakeLlm(["ignored"], True),
        decision=routing.RouteDecision(route="unsupported"))
    assert not answer.generated
    assert not answer.data_available
    assert "Structured questions I can answer:" in answer.text
    assert "Total sales revenue" in answer.text


# --------------------------------------------------------------------- checks


def test_verify_figures_accepts_copies_and_copies_rounded_to_cents():
    evidence = "amount=500.894 other=1000000"
    assert verify_figures("The amount is 500.894.", evidence) == []
    assert verify_figures("The amount is 1,000,000.", evidence) == []
    assert verify_figures("The amount is 500.89 dollars.", evidence) == []


def test_verify_figures_names_what_the_evidence_does_not_contain():
    assert verify_figures("Profit is 777.77 today.", "revenue=1000") == ["777.77"]


def test_small_integers_and_dates_are_not_business_figures():
    answer = "As of 2023-01-15, 3 items moved and 1050 cars shipped."
    assert verify_figures(answer, "1050 present") == []
    # a figure with a decimal part is always considered, even a threshold
    assert verify_figures("The threshold is 1.5 days.", "") == ["1.5"]


def test_the_answer_payload_carries_every_guarantee_field(settings):
    answer = answer_question(
        "How many pending purchase orders are there?", settings,
        FakeLlm([], available=False), decision=pending_decision(),
        connection=pending_database())
    body = answer.to_dict()
    for key in ("answer", "route", "classifier", "query", "sql", "citations",
                "data_available", "documents_available", "generated", "model",
                "refusals", "warnings", "unverified_figures"):
        assert key in body
    assert body["sql"]["row_count"] == 1
    assert body["sql"]["rows"] == [{"pending_orders": 225}]


# --------------------------------------------------------------- connections


def test_open_connection_reports_a_dead_server_with_a_redacted_dsn(monkeypatch):
    monkeypatch.setenv("POSTGRES_PASSWORD", "s3cret-value")
    monkeypatch.setattr(db_config, "connect",
                        lambda cfg: (_ for _ in ()).throw(
                            ConnectionError("refused")))
    connection, error, opened = open_connection(make_settings(None))
    assert connection is None and not opened
    assert "ConnectionError" in error
    assert "***redacted***" in error
    assert "s3cret-value" not in error


def test_open_connection_reports_a_missing_credential(monkeypatch):
    def refuse():
        raise RuntimeError("POSTGRES_PASSWORD is not set")

    monkeypatch.setattr(db_config, "config_from_env", refuse)
    connection, error, opened = open_connection(make_settings(None))
    assert connection is None
    assert "POSTGRES_PASSWORD is not set" in error
