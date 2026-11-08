"""The SQL allowlist: only approved names run, limits are clamped before SQL.

The unit tests use a FakeDatabase, so the contract is checked without a
server. The last two tests are the read-only complement to
tests/test_db_postgres.py: real SELECTs against the live `erp` schema,
always in a transaction that is rolled back, skipped whenever the
POSTGRES_* settings do not reach a server. No schema or data is touched.
"""
from datetime import date
from decimal import Decimal

import pytest

from src.ai import sql_tools
from src.db import analytics
from src.db import config as db_config

from .ai_fakes import FakeDatabase

REVENUE_ROW = [{
    "orders": 31465, "order_merchandise": "109846381.4039",
    "order_tax": "10186974.4602", "order_freight": "3183430.2518",
    "order_total_due": "123216786.1159", "order_lines": 121317,
    "line_revenue": "109846381.399888", "units_sold": 274914,
}]
PRODUCT_ROWS = [
    {"product_id": 782, "name": "Mountain-200 Black, 38",
     "revenue": "4400592.800400", "units_sold": 2977},
    {"product_id": 718, "name": "Mountain-100 Silver, 44",
     "revenue": "3056781.480100", "units_sold": 2394},
]
PENDING_ROWS = [{"pending_orders": 225}]


def database() -> FakeDatabase:
    return FakeDatabase([
        ("order_total_due", REVENUE_ROW),
        ("revenue", PRODUCT_ROWS),
        ("pending_orders", PENDING_ROWS),
    ])


# ----------------------------------------------------------------- clamping


def test_the_limit_is_clamped_to_the_ceiling():
    assert sql_tools.clamp_limit(200, 50) == 50
    assert sql_tools.clamp_limit(5, 50) == 5


def test_the_limit_accepts_the_string_a_json_payload_may_carry():
    assert sql_tools.clamp_limit("10", 50) == 10


def test_the_limit_refuses_to_guess():
    for value in ("ten", None, [], 10.5, True, 0, -3):
        with pytest.raises(ValueError):
            sql_tools.clamp_limit(value, 50)


# ------------------------------------------------------------ the allowlist


def test_only_approved_names_reach_sql():
    with pytest.raises(sql_tools.UnknownQuery):
        sql_tools.execute(database(), "SELECT count(*) FROM erp.product")
    with pytest.raises(sql_tools.UnknownQuery):
        sql_tools.get("top_customers_by_country")


def test_the_allowlist_mirrors_the_eight_phase_two_questions():
    assert sql_tools.tool_names() == analytics.QUESTIONS
    assert len(sql_tools.ALLOWLIST) == 8


def test_the_catalog_tells_the_classifier_what_exists_and_no_sql():
    text = sql_tools.catalog()
    for name in sql_tools.tool_names():
        assert name in text
    assert "SELECT" not in text.upper()
    for name in sql_tools.tool_names():
        tool = sql_tools.ALLOWLIST[name]
        marker = ("parameters: none" if not tool.params
                  else f"parameters: {', '.join(tool.params)}")
        assert marker in text


def test_titles_are_reported_in_brief_order():
    assert sql_tools.catalogue_titles()[:2] == [
        "Total sales revenue", "Top products by revenue"]


# --------------------------------------------------------------- execution


def test_an_approved_query_runs_and_reports_its_metadata():
    result = sql_tools.execute(database(), "total_sales_revenue")
    assert result.name == "total_sales_revenue"
    assert result.grain == "order and line, reported separately"
    assert result.row_count == 1
    assert result.columns[0] == "orders"
    assert result.rows[0]["order_total_due"] == "123216786.1159"
    assert result.parameters == {}


def test_the_limit_parameter_is_bound_not_interpolated():
    conn = database()
    result = sql_tools.execute(conn, "top_products_by_revenue", limit=2,
                               limit_ceiling=50)
    executed_sql, bound = conn.executed[0]
    assert bound == (2,)
    assert "LIMIT  %s" in executed_sql
    assert result.rows == PRODUCT_ROWS


def test_a_limit_above_the_ceiling_is_clamped_not_dropped():
    conn = database()
    result = sql_tools.execute(conn, "top_products_by_revenue", limit=900,
                               limit_ceiling=50)
    assert conn.executed[0][1] == (50,)
    assert result.parameters == {"limit": 50}


def test_a_parameter_the_query_does_not_accept_is_a_refusal():
    with pytest.raises(ValueError):
        sql_tools.execute(database(), "pending_purchase_order_count",
                          parameters={"sort": "desc"})


def test_the_pending_count_stays_a_named_column():
    result = sql_tools.execute(database(), "pending_purchase_order_count")
    assert result.columns == ["pending_orders"]
    assert result.rows == [{"pending_orders": 225}]


# ------------------------------------------------------------- normalisation


def test_decimals_stay_text_and_dates_stay_isoformatted():
    assert sql_tools.cell(Decimal("123216786.1159")) == "123216786.1159"
    assert sql_tools.cell(date(2024, 1, 5)) == "2024-01-05"
    assert sql_tools.cell(None) is None


def test_a_tool_returning_an_unnamed_number_is_a_programming_error():
    with pytest.raises(TypeError):
        sql_tools._as_rows(225)


def test_rows_are_normalised_from_dicts_and_lists():
    assert sql_tools._as_rows({"a": 1}) == [{"a": 1}]
    assert sql_tools._as_rows([{"a": 1}, {"a": 2}]) == [{"a": 1}, {"a": 2}]


# --------------------------------------------------------------- rendering


def test_an_empty_result_says_there_is_no_figure_to_report():
    executed = sql_tools.execute(FakeDatabase([("revenue", [])]),
                                 "top_products_by_revenue")
    assert executed.rows == []
    assert "no rows" in executed.render(max_rows=10)


def test_rendered_rows_carry_the_grain_and_the_count():
    executed = sql_tools.execute(database(), "total_sales_revenue")
    text = executed.render(max_rows=15)
    assert text.splitlines()[0] == (
        "Total sales revenue [order and line, reported separately grain]"
        " - 1 rows returned")
    assert "order_total_due=123216786.1159" in text
    assert "note: order_total_due is order grain" in text


def test_truncation_is_declared_in_both_the_text_and_the_payload():
    tools = sql_tools.execute(database(), "top_products_by_revenue", limit=2,
                              limit_ceiling=50)
    text = tools.render(max_rows=1)
    assert "... 1 more rows not shown" in text
    assert "rows returned" in text.splitlines()[0]

    body = tools.to_dict(max_rows=1)
    assert body["row_count"] == 2
    assert body["returned_rows"] == 1
    assert body["truncated"] is True
    assert len(body["rows"]) == 1


# ------------------------------------- read-only live checks, skip w/o server


def _live_connection():
    reachable, reason = db_config.server_reachable()
    if not reachable:
        pytest.skip(f"live PostgreSQL unavailable: {reason}")
    return db_config.connect()


def test_live_total_revenue_matches_the_published_totals():
    conn = _live_connection()
    try:
        result = sql_tools.execute(conn, "total_sales_revenue")
        row = result.rows[0]
        assert result.row_count == 1
        assert row["orders"] == 31465
        assert row["order_lines"] == 121317
        assert str(row["order_total_due"]).startswith("123216786.1159")
    finally:
        conn.rollback()
        conn.close()


def test_live_pending_count_is_at_order_grain():
    conn = _live_connection()
    try:
        result = sql_tools.execute(conn, "pending_purchase_order_count")
        assert result.rows[0]["pending_orders"] == 225
    finally:
        conn.rollback()
        conn.close()
