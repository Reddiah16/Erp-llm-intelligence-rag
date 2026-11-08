"""The eight business questions, as SQL against the erp schema.

This module is the read side of the migration: it holds no pandas, no
business rules and no reformatting of the data. Each question is one named SQL
constant, so a reviewer can copy it into psql unchanged, and one thin function
that runs it. Nothing here recomputes what the schema already guarantees.

Order-level and line-level money stay separate in every query, which is the
reason the schema was normalized in the first place:

  order grain   erp.sales_order.total_due     one value per order
  line grain    erp.sales_order_line.line_total  one value per detail row

A single SELECT never sums both, and a header-to-line join only ever appears
where a line-grain figure is being aggregated. Summing total_due across a
joined frame inflates it by about 23.75x on this dataset (average 3.856 lines
per order), and the flat CSV pipeline had exactly one defence against that:
developer discipline. Here the GROUP BY grain is the defence.

    python -m src.db.analytics            # answer all eight
    python -m src.db.analytics --json     # machine-readable output
"""
import argparse
import json
import sys
from decimal import Decimal
from typing import Any, Dict, List, Optional

# ----------------------------------------------------------------- question 1
# Two separate scalar subqueries rather than one join: joining lines would
# multiply every order-level column by its line count.
TOTAL_SALES_REVENUE_SQL = """
SELECT (SELECT count(*)                    FROM erp.sales_order)        AS orders,
       (SELECT sum(sub_total)              FROM erp.sales_order)        AS order_merchandise,
       (SELECT sum(tax_amt)                FROM erp.sales_order)        AS order_tax,
       (SELECT sum(freight)                FROM erp.sales_order)        AS order_freight,
       (SELECT sum(total_due)              FROM erp.sales_order)        AS order_total_due,
       (SELECT count(*)                    FROM erp.sales_order_line)   AS order_lines,
       (SELECT sum(line_total)             FROM erp.sales_order_line)   AS line_revenue,
       (SELECT sum(order_qty)              FROM erp.sales_order_line)   AS units_sold
"""

# ----------------------------------------------------------------- question 2
# Product revenue is a LINE-grain figure: a product only ever appears on
# detail rows. total_due is not selected here for exactly that reason.
TOP_PRODUCTS_BY_REVENUE_SQL = """
SELECT p.product_id,
       p.name,
       p.product_number,
       sum(l.line_total)            AS revenue,
       sum(l.order_qty)             AS units_sold,
       count(*)                     AS order_lines,
       count(DISTINCT l.sales_order_id) AS orders
FROM   erp.sales_order_line l
JOIN   erp.product p USING (product_id)
GROUP  BY p.product_id, p.name, p.product_number
ORDER  BY revenue DESC, p.product_id
LIMIT  %s
"""

# ----------------------------------------------------------------- question 3
# Customer revenue is ORDER-grain: one order belongs to one customer, so no
# detail line is touched and no amount can be repeated.
TOP_CUSTOMERS_BY_REVENUE_SQL = """
SELECT c.customer_id,
       c.account_number,
       count(o.sales_order_id)   AS orders,
       sum(o.total_due)          AS revenue,
       min(o.order_date)         AS first_order_date,
       max(o.order_date)         AS last_order_date
FROM   erp.sales_order o
JOIN   erp.customer c USING (customer_id)
GROUP  BY c.customer_id, c.account_number
ORDER  BY revenue DESC, c.customer_id
LIMIT  %s
"""

# ----------------------------------------------------------------- question 4
# From the view, so the "no inventory record is not zero" rule is applied in
# one place instead of being re-typed per query.
# The two stock questions return every row: 69 and 7 on the canonical dataset,
# small enough to hand over whole and large enough that the caller decides what
# to display. No LIMIT parameter means no way to silently truncate an alert.
PRODUCTS_BELOW_SAFETY_STOCK_SQL = """
SELECT product_id, name, product_number,
       safety_stock_level, total_quantity,
       safety_stock_level - total_quantity AS shortfall
FROM   erp.stock_position
WHERE  below_safety
ORDER  BY shortfall DESC, product_id
"""

# ----------------------------------------------------------------- question 5
PRODUCTS_NEEDING_REORDER_SQL = """
SELECT product_id, name, product_number,
       reorder_point, total_quantity,
       reorder_point - total_quantity AS shortfall
FROM   erp.stock_position
WHERE  needs_reorder
ORDER  BY shortfall DESC, product_id
"""

# ----------------------------------------------------------------- question 6
# Order grain: 225 pending orders, whose 477 detail lines are a different
# number and not the answer to this question.
PENDING_PURCHASE_ORDER_COUNT_SQL = """
SELECT count(*) AS pending_orders
FROM   erp.purchase_order
WHERE  order_status = 1          -- 1 Pending, per loader.PurchaseOrderHeader
"""

# ----------------------------------------------------------------- question 7
PENDING_PURCHASE_ORDER_VALUE_SQL = """
SELECT count(*)            AS pending_orders,
       sum(sub_total)      AS pending_merchandise,
       sum(total_due)      AS pending_total_due
FROM   erp.purchase_order
WHERE  order_status = 1
"""

# ----------------------------------------------------------------- question 8
# Both grains per month, but each aggregated in its own CTE and only the
# finished aggregates are joined - on the month key, never on the rows.
#
# The month is cast back to DATE: date_trunc() on a DATE answers with a
# timestamptz, which would make the bucket depend on the server's timezone
# setting. A calendar month is a date, not a moment.
MONTHLY_SALES_TRENDS_SQL = """
WITH orders AS (
    SELECT date_trunc('month', order_date::timestamp)::date AS month,
           count(*)                                         AS orders,
           sum(sub_total)                                   AS order_merchandise,
           sum(total_due)                                   AS order_total_due
    FROM   erp.sales_order
    GROUP  BY 1
), lines AS (
    SELECT date_trunc('month', o.order_date::timestamp)::date AS month,
           count(*)                                           AS order_lines,
           sum(l.line_total)                                  AS line_revenue
    FROM   erp.sales_order o
    JOIN   erp.sales_order_line l USING (sales_order_id)
    GROUP  BY 1
)
SELECT o.month, o.orders, l.order_lines,
       o.order_merchandise, l.line_revenue, o.order_total_due,
       l.line_revenue - lag(l.line_revenue) OVER (ORDER BY o.month)
           AS line_revenue_change
FROM   orders o
JOIN   lines  l USING (month)
ORDER  BY o.month
"""

# Every question in the order the brief lists them. The CLI walks this.
QUESTIONS: List[str] = [
    "total_sales_revenue",
    "top_products_by_revenue",
    "top_customers_by_revenue",
    "products_below_safety_stock",
    "products_needing_reorder",
    "pending_purchase_order_count",
    "pending_purchase_order_value",
    "monthly_sales_trends",
]

DEFAULT_LIMIT = 10


def _rows(conn, sql: str, params=None) -> List[Dict[str, Any]]:
    """Rows as dicts, keyed by the column names PostgreSQL reported.

    psycopg 3 returns plain tuples; the names come from the cursor description
    so every alias written in the SQL above is also the key the caller reads.
    """
    result = conn.execute(sql, params)
    names = [column.name for column in result.description]
    return [dict(zip(names, row)) for row in result.fetchall()]


# -------------------------------------------------------------- query functions

def total_sales_revenue(conn) -> Dict[str, Any]:
    """Question 1. Order-level and line-level totals, side by side."""
    return _rows(conn, TOTAL_SALES_REVENUE_SQL)[0]


def top_products_by_revenue(conn, limit: int = DEFAULT_LIMIT) -> List[Dict[str, Any]]:
    """Question 2. Line-grain revenue per product."""
    return _rows(conn, TOP_PRODUCTS_BY_REVENUE_SQL, (limit,))


def top_customers_by_revenue(conn, limit: int = DEFAULT_LIMIT) -> List[Dict[str, Any]]:
    """Question 3. Order-grain revenue per customer."""
    return _rows(conn, TOP_CUSTOMERS_BY_REVENUE_SQL, (limit,))


def products_below_safety_stock(conn) -> List[Dict[str, Any]]:
    """Question 4. Recorded stock only; unrecorded products are unknown, not low."""
    return _rows(conn, PRODUCTS_BELOW_SAFETY_STOCK_SQL)


def products_needing_reorder(conn) -> List[Dict[str, Any]]:
    """Question 5. A subset of the below-safety set on this dataset."""
    return _rows(conn, PRODUCTS_NEEDING_REORDER_SQL)


def pending_purchase_order_count(conn) -> int:
    """Question 6. Purchase orders, not purchase order lines."""
    return _rows(conn, PENDING_PURCHASE_ORDER_COUNT_SQL)[0]["pending_orders"]


def pending_purchase_order_value(conn) -> Dict[str, Any]:
    """Question 7. Order-grain value of the same 225 pending orders."""
    return _rows(conn, PENDING_PURCHASE_ORDER_VALUE_SQL)[0]


def monthly_sales_trends(conn) -> List[Dict[str, Any]]:
    """Question 8. 38 months, order grain and line grain in separate columns."""
    return _rows(conn, MONTHLY_SALES_TRENDS_SQL)


def answer_all(conn, limit: int = DEFAULT_LIMIT) -> Dict[str, Any]:
    """All eight answers in one call, keyed by the question name."""
    return {
        "total_sales_revenue":            total_sales_revenue(conn),
        "top_products_by_revenue":        top_products_by_revenue(conn, limit),
        "top_customers_by_revenue":       top_customers_by_revenue(conn, limit),
        "products_below_safety_stock":    products_below_safety_stock(conn),
        "products_needing_reorder":       products_needing_reorder(conn),
        "pending_purchase_order_count":   pending_purchase_order_count(conn),
        "pending_purchase_order_value":   pending_purchase_order_value(conn),
        "monthly_sales_trends":           monthly_sales_trends(conn),
    }


# -------------------------------------------------------------------- printing

TITLES = {
    "total_sales_revenue":          "1. Total sales revenue",
    "top_products_by_revenue":      "2. Top products by revenue",
    "top_customers_by_revenue":     "3. Highest-revenue customers",
    "products_below_safety_stock":  "4. Products below safety stock",
    "products_needing_reorder":     "5. Products needing reorder",
    "pending_purchase_order_count": "6. Pending purchase order count",
    "pending_purchase_order_value": "7. Total value of pending purchase orders",
    "monthly_sales_trends":         "8. Monthly sales trends",
}


def _jsonable(value):
    if isinstance(value, Decimal):
        return str(value)          # str keeps every stored decimal, float would not
    return value


def format_results(results: Dict[str, Any], max_rows: int = 50) -> str:
    """Plain-text report. Money prints at full stored precision.

    The blocks come in the order of QUESTIONS but are driven by what `results`
    actually holds, so `--question` can print a single answer instead of
    reaching for the seven that were not asked for. A key that is not a known
    question is a caller bug, and saying so beats an empty report.
    """
    unknown = set(results) - set(QUESTIONS)
    if unknown:
        raise ValueError(f"not a known question: {sorted(unknown)}")
    blocks = []
    for key in [question for question in QUESTIONS if question in results]:
        value = results[key]
        lines = [TITLES[key]]
        if isinstance(value, list):
            for row in value[:max_rows]:
                cells = " ".join(f"{column}={_jsonable(cell)}"
                                 for column, cell in row.items())
                lines.append(f"  {cells}")
            if len(value) > max_rows:
                lines.append(f"  ... {len(value) - max_rows} more rows")
        elif isinstance(value, dict):
            for column, cell in value.items():
                lines.append(f"  {column:24s} {_jsonable(cell)}")
        else:                       # a single number, like the pending order count
            lines.append(f"  {_jsonable(value)}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.db.analytics",
        description="Run the eight business questions against the erp schema.")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT,
                        help="rows for the ranked questions (default 10)")
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument("--question", choices=QUESTIONS,
                        help="answer only one question")
    args = parser.parse_args(argv)

    from src.db.config import config_from_env, connect
    try:
        config = config_from_env()
    except RuntimeError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    try:
        conn = connect(config)
    except Exception as exc:                       # noqa: BLE001 - reported below
        print(f"could not connect to PostgreSQL: {type(exc).__name__}: {exc}\n"
              f"target was {config.safe_dsn}", file=sys.stderr)
        return 2

    try:
        results = answer_all(conn, limit=args.limit)
        if args.question:
            results = {args.question: results[args.question]}
        if args.as_json:
            print(json.dumps(results, default=str, indent=2, sort_keys=False))
        else:
            print(format_results(results))
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
