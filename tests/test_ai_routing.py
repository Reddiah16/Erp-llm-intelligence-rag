"""Routing: classify a question, validate the payload, refuse what is not allowed.

The model is always a fake; what is asserted here is the contract around
it, which is where the security property lives: only names in
sql_tools.ALLOWLIST can survive, model text never reaches SQL, and every
downgrade is reported in the decision, not swallowed.
"""
import pytest

from src.ai.llm import LlmProtocolError
from src.ai.routing import (GUARDED_MESSAGES, ROUTES, classify, heuristic,
                            parse_classification, validate)
from src.llm.prompts import ROUTE_PROMPT

from .ai_fakes import FakeLlm


def labelled(route: str, query: str = "NONE", limit: str = "NONE",
             reason: str = "justified") -> str:
    """The exact four-line body the ROUTE_PROMPT demands."""
    return (f"ROUTE: {route}\nQUERY: {query}\nLIMIT: {limit}\n"
            f"REASON: {reason}")


def classified_answer(route: str, query: str = "NONE", limit: str = "NONE",
                      reason: str = "justified") -> FakeLlm:
    return FakeLlm([labelled(route, query, limit, reason)], True)


# --------------------------------------------------------------- parsing


def test_the_label_block_parses_into_a_payload():
    payload = parse_classification(labelled("sql", "total_sales_revenue",
                                            limit="5"))
    assert payload["route"] == "sql"
    assert payload["query"] == "total_sales_revenue"
    assert payload["params"] == {"limit": "5"}


def test_empty_query_and_limit_values_are_noted_as_missing():
    payload = parse_classification(labelled("rag"))
    assert payload["query"] is None
    assert "params" not in payload


def test_a_reply_without_a_route_line_is_a_protocol_failure():
    with pytest.raises(LlmProtocolError):
        parse_classification("QUERY: total_sales_revenue\nREASON: none")


def test_case_insensitive_labels_and_junk_fences_are_tolerated():
    payload = parse_classification("```sql\nroute: RAG\nquery: none\n"
                                   "limit: none\nreason: policy question")
    assert payload["route"] == "rag"


# --------------------------------------------------------------- validation


def test_an_approved_query_on_the_sql_route_executes():
    decision = validate(parse_classification(labelled("sql",
                                                      "total_sales_revenue")))
    assert decision.route == "sql"
    assert decision.executable
    assert decision.needs_sql and not decision.needs_documents


def test_a_limit_parameter_survives_validation():
    decision = validate(parse_classification(labelled("sql",
                                                      "top_products_by_revenue",
                                                      limit="3")))
    assert decision.params == {"limit": "3"}
    assert decision.executable


def test_a_parameter_the_query_does_not_accept_is_refused():
    payload = parse_classification(labelled("sql", "top_products_by_revenue"))
    payload["params"] = {"sort": "asc"}
    decision = validate(payload)
    assert decision.executable and not decision.params
    assert any("does not accept parameter" in item
               for item in decision.refusals)


def test_params_that_are_not_an_object_are_dropped():
    payload = parse_classification(labelled("sql", "top_products_by_revenue"))
    payload["params"] = "top=3; DROP TABLE erp.product"
    decision = validate(payload)
    assert decision.executable
    assert any("params" in item for item in decision.refusals)


def test_an_unrecognised_route_with_an_approved_query_is_read_as_sql():
    decision = validate(parse_classification(labelled("figures",
                                                      "total_sales_revenue")))
    assert decision.route == "sql"
    assert decision.executable
    assert any("unrecognised route" in item for item in decision.refusals)


def test_an_unrecognised_route_without_a_query_is_unsupported():
    decision = validate(parse_classification("ROUTE: custom\nQUERY: none\n"
                                             "REASON: it seemed apt"))
    assert decision.route == "unsupported"
    assert decision.classifier == "llm"
    assert not decision.executable


def test_model_sql_is_never_executed():
    decision = validate(parse_classification(
        labelled("sql", "SELECT product_name FROM erp.product")))
    assert decision.route == "unsupported"
    assert decision.query is None
    assert any("never executed" in item for item in decision.refusals)


def test_an_unapproved_plain_text_query_is_refused():
    decision = validate(parse_classification(
        labelled("sql", "top_customers_by_city")))
    assert decision.route == "unsupported"
    assert any("is not an approved query" in item for item in decision.refusals)


def test_a_hybrid_route_without_a_query_becomes_a_document_question():
    decision = validate(parse_classification(labelled("hybrid")))
    assert decision.route == "rag"
    assert decision.needs_documents and not decision.needs_sql
    assert any("became a document question" in item
               for item in decision.refusals)


def test_a_hybrid_route_with_a_bad_query_becomes_a_document_question():
    decision = validate(parse_classification(labelled("hybrid",
                                                      "top_customers_by_city")))
    assert decision.route == "rag"
    assert any("not an approved query" in item for item in decision.refusals)


def test_a_hybrid_route_with_an_approved_query_stays_hybrid():
    decision = validate(parse_classification(labelled("hybrid",
                                                      "total_sales_revenue")))
    assert decision.route == "hybrid"
    assert decision.needs_sql and decision.needs_documents


def test_sql_in_a_hybrid_route_is_refused_outright():
    decision = validate(parse_classification(
        labelled("hybrid", "select * from sales")))
    assert decision.route == "unsupported"
    assert decision.query is None


def test_a_query_cannot_reach_a_route_that_reads_no_database():
    decision = validate(parse_classification(labelled("rag",
                                                      "total_sales_revenue")))
    assert decision.route == "rag"
    assert decision.query is None
    assert any("ignored" in item for item in decision.refusals)


def test_an_empty_limit_is_not_sent_as_a_parameter():
    decision = validate(parse_classification(labelled("sql",
                                                      "top_products_by_revenue",
                                                      limit="NONE")))
    assert decision.params == {}
    assert decision.executable


def test_the_reason_is_reported_but_bounded():
    decision = validate(parse_classification(labelled("sql",
                                                      "total_sales_revenue",
                                                      reason="word " * 400)))
    assert len(decision.reason) <= 300


def test_every_route_value_is_knowable():
    assert set(ROUTES) == {"sql", "rag", "hybrid", "unsupported"}


# --------------------------------------------------------------- heuristic


def test_a_figure_question_routes_to_sql():
    decision = heuristic("How much total revenue did we record?")
    assert decision.route == "sql"
    assert decision.query == "total_sales_revenue"
    assert decision.classifier == "heuristic"


def test_the_specific_pending_order_marker_wins_over_the_generic_one():
    assert heuristic("How many pending purchase orders are open?").query == (
        "pending_purchase_order_count")
    assert heuristic("How many open purchase orders are there?").query == (
        "pending_purchase_order_count")
    # a bare "purchase order" question is value order-grain by rule, never
    # silently read as the count: the two are different questions.
    assert heuristic("What is the value of purchase orders outstanding?").query == (
        "pending_purchase_order_value")


def test_a_policy_question_routes_to_the_documents():
    decision = heuristic("What is the definition of the data dictionary "
                         "document?")
    assert decision.route == "rag"
    assert decision.query is None


def test_a_figure_and_a_policy_marker_together_select_hybrid():
    decision = heuristic("What is the rule for pending purchase orders, "
                         "and how many are open?")
    assert decision.route == "hybrid"
    assert decision.query == "pending_purchase_order_count"


def test_a_figure_question_with_no_matching_query_is_unsupported():
    decision = heuristic("How many sales regions are there, and what is a "
                         "region number count?")
    assert decision.route == "unsupported"
    assert "no approved query" in decision.reason


def test_small_talk_sends_nothing_anywhere():
    decision = heuristic("Tell me a joke about the weather in Iceland")
    assert decision.route == "unsupported"


def test_the_month_marker_is_more_specific_than_the_revenue_marker():
    decision = heuristic("List revenue by month")
    assert decision.query == "monthly_sales_trends"


# --------------------------------------------------------------- classify


def test_a_model_reply_is_validated_not_trusted():
    decision = classify("Which products are below safety stock?", None,
                        FakeLlm([labelled("sql", "products_below_safety_stock")],
                                True))
    assert decision.route == "sql"
    assert decision.query == "products_below_safety_stock"
    assert decision.classifier == "llm"


def test_explicit_mixed_intent_overrides_a_model_rag_route():
    question = ("Explain the procurement guidelines and identify products "
                "that need reordering.")
    decision = classify(question, None, FakeLlm([labelled("rag")], True))
    assert decision.route == "hybrid"
    assert decision.query == "products_needing_reorder"
    assert decision.needs_sql and decision.needs_documents
    assert decision.classifier == "heuristic"
    assert any("two-part intent" in item for item in decision.refusals)


def test_a_standalone_reorder_definition_stays_on_the_document_route():
    decision = classify("What is the reorder point?", None,
                        FakeLlm([labelled("rag")], True))
    assert decision.route == "rag"
    assert not decision.needs_sql


def test_an_explicit_uploaded_document_question_uses_retrieval():
    decision = classify(
        "Summarize the QA validation policy document that was uploaded.",
        None, FakeLlm([labelled("unsupported")], True))
    assert decision.route == "rag"
    assert decision.needs_documents
    assert not decision.needs_sql
    assert decision.classifier == "heuristic"


def test_exact_return_period_question_routes_to_rag_despite_a_fake_query():
    question = ("According to customer_return_refund_policy.md, how many "
                "calendar days do customers have to request a return?")
    decision = classify(
        question, None,
        FakeLlm([labelled("sql", "pending_return_request_days")], True))
    assert decision.route == "rag"
    assert decision.query is None
    assert decision.needs_documents and not decision.needs_sql


def test_a_named_corpus_filename_overrides_an_unrelated_approved_query():
    decision = classify(
        "According to customer_return_refund_policy.md, what is the return "
        "eligibility rule?",
        None, FakeLlm([labelled("sql", "total_sales_revenue")], True))
    assert decision.route == "rag"
    assert decision.query is None
    assert decision.needs_documents and not decision.needs_sql


@pytest.mark.parametrize("question", [
    "How many business days does refund approval take?",
    "Are shipping charges refundable when a product arrives damaged?",
])
def test_return_policy_questions_without_a_filename_route_to_rag(question):
    decision = classify(
        question, None,
        FakeLlm([labelled("sql", "pending_purchase_order_count")], True))
    assert decision.route == "rag"
    assert decision.query is None
    assert decision.needs_documents and not decision.needs_sql


@pytest.mark.parametrize("question", [
    "What is the inventory safety stock policy?",
    "Explain the procurement guidelines.",
])
def test_inventory_and_procurement_policy_questions_do_not_run_sql(question):
    decision = classify(
        question, None,
        FakeLlm([labelled("sql", "total_sales_revenue")], True))
    assert decision.route == "rag"
    assert decision.query is None
    assert decision.needs_documents and not decision.needs_sql


def test_an_unrelated_figure_cannot_use_an_approved_but_unrelated_query():
    decision = classify(
        "How many employees work in the finance department?",
        None, FakeLlm([labelled("sql", "pending_purchase_order_count")], True))
    assert decision.route == "unsupported"
    assert decision.query is None
    assert not decision.needs_sql


def test_a_garbage_model_reply_falls_back_to_the_heuristics():
    decision = classify("How much total revenue is recorded?", None,
                        FakeLlm(["I am just a helpful assistant"], True))
    assert decision.route == "sql"
    assert decision.classifier == "heuristic"
    assert any("classifier failed" in item for item in decision.refusals)


def test_an_unavailable_model_falls_back_to_the_heuristics():
    decision = classify("How much total revenue is recorded?", None,
                        FakeLlm([], available=False))
    assert decision.route == "sql"
    assert decision.classifier == "heuristic"
    assert any("classifier unavailable" in item for item in decision.refusals)


def test_an_empty_question_never_reaches_the_model():
    decision = classify("   ", None, FakeLlm([labelled("sql")], True))
    assert decision.route == "unsupported"
    assert decision.reason == GUARDED_MESSAGES["empty"]
    assert decision.classifier == "guard"


def test_an_oversized_question_is_refused_at_the_guard():
    long_question = "revenue " * 200          # far beyond 500 characters
    decision = classify(long_question, None, FakeLlm([labelled("sql")], True))
    assert decision.route == "unsupported"
    assert "500-character" in decision.reason or "character limit" in decision.reason


# ------------------------------------------- live-routing regression (QA 2026-11)

def test_the_route_prompt_carries_the_list_vs_definition_examples():
    """The shipped prompt must disambiguate the two 'reorder' wordings.

    QA on 2026-11 measured the shipped prompt misclassifying 'Which products
    need reordering?' in 8 of 8 live runs (the product list never ran: rag
    definition, and top_products instead of the reorder list in a session).
    The examples block in ROUTE_PROMPT is the measured fix: 22 of 24 live
    classification runs across the twelve canonical questions matched the
    expected route and query, and the one deviation kept the correct approved
    query with both evidence sources. This test pins the examples so a later
    prompt edit cannot silently remove the measured fix.
    """
    template = ROUTE_PROMPT.template
    assert "Which products need reordering?" in template, \
        "the list question needs its sql example, or the 1.5B route model " \
        "answers it with a rag definition again"
    assert "what is the reorder point?" in template, \
        "the definition question must stay rag - a definition gets no query"
    assert "Explain the guidelines and identify the products" in template, \
        "the compound question must keep its hybrid example"
    collapsed = " ".join(template.split())
    assert "A standalone list question is never hybrid" in collapsed, \
        "the hybrid route keeps a standalone list on the sql route"
    assert "names a .md or .txt policy/document" in collapsed, \
        "explicit document references must select retrieval"


def test_a_reorder_list_payload_executes_the_reorder_query():
    """The payload the route model produces for the list question runs it."""
    decision = validate(parse_classification(labelled(
        "sql", "products_needing_reorder", reason="requested list")))
    assert decision.route == "sql"
    assert decision.query == "products_needing_reorder"
    assert decision.executable


def test_a_reorder_definition_payload_never_touches_the_database():
    decision = validate(parse_classification(labelled("rag", "NONE",
                                                     reason="definition")))
    assert decision.route == "rag"
    assert decision.query is None
    assert not decision.executable


def test_a_compound_payload_reaches_both_evidence_sources():
    decision = validate(parse_classification(labelled(
        "hybrid", "products_needing_reorder", reason="rule and list")))
    assert decision.route == "hybrid"
    assert decision.query == "products_needing_reorder"
    assert decision.needs_documents and decision.needs_sql
