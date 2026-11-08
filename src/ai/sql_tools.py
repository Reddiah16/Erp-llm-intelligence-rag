"""The only structured data the assistant may read: an explicit allowlist.

Eight verified Phase 2 queries, wrapped with the metadata a router needs to
choose one and the formatting a caller needs to display one. Nothing in this
module accepts SQL text. `execute` takes a query NAME, looks it up in ALLOWLIST
and calls the function registered there, so a name that is not listed cannot be
run and there is no code path anywhere that hands a string to the database as
if it were a query. That is the barrier. looks_like_sql() in src/ai/llm.py only
records a refusal so the response says why.

Each tool declares the GRAIN its money columns live at, and the grain note
travels with the result into the model's context and into the API response.
Order-level and line-level figures were the reason Phase 2 normalised the
schema, so the separation is kept here rather than reconstructed by whoever
formats the answer.
"""
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from src.db import analytics


class UnknownQuery(KeyError):
    """A query name that is not on the allowlist."""


@dataclass(frozen=True)
class SqlTool:
    name: str
    title: str
    description: str
    grain: str
    grain_note: str
    run: Callable[..., Any]
    params: Tuple[str, ...] = ()


def _pending_purchase_order_count(conn) -> Dict[str, Any]:
    """The analytics function returns a bare integer.

    Wrapping it in a named column keeps the result self-describing all the way
    to the response body, where "value=225" would leave a reader to guess what
    was counted.
    """
    return {"pending_orders": analytics.pending_purchase_order_count(conn)}


ALLOWLIST: Dict[str, SqlTool] = {
    "total_sales_revenue": SqlTool(
        name="total_sales_revenue",
        title="Total sales revenue",
        description="Order counts, order-level subtotal/tax/freight/total due, "
                    "line counts, line-level revenue and units sold, in one row.",
        grain="order and line, reported separately",
        grain_note="order_total_due is order grain; line_revenue is line grain. "
                   "They measure different things and must not be added.",
        run=analytics.total_sales_revenue,
    ),
    "top_products_by_revenue": SqlTool(
        name="top_products_by_revenue",
        title="Top products by revenue",
        description="Highest-selling products by line-level revenue, with units "
                    "sold and order counts. Parameter: limit.",
        grain="line",
        grain_note="Product revenue is line grain (LineTotal). Order-level "
                   "TotalDue cannot be attributed to a product and is absent.",
        run=analytics.top_products_by_revenue,
        params=("limit",),
    ),
    "top_customers_by_revenue": SqlTool(
        name="top_customers_by_revenue",
        title="Top customers by revenue",
        description="Customers contributing the most order-level total due, with "
                    "order counts and first and last order dates. Parameter: limit.",
        grain="order",
        grain_note="Customer revenue is order grain: one order, one customer, "
                   "no line is touched, so nothing can repeat.",
        run=analytics.top_customers_by_revenue,
        params=("limit",),
    ),
    "products_below_safety_stock": SqlTool(
        name="products_below_safety_stock",
        title="Products below safety stock",
        description="Every product whose recorded on-hand quantity is below its "
                    "safety stock level, worst shortfall first.",
        grain="inventory",
        grain_note="Products with no inventory record are absent: unknown "
                   "quantity is not zero quantity, so they are not shortages.",
        run=analytics.products_below_safety_stock,
    ),
    "products_needing_reorder": SqlTool(
        name="products_needing_reorder",
        title="Products needing reorder",
        description="Every product whose recorded on-hand quantity is below its "
                    "reorder point, worst shortfall first.",
        grain="inventory",
        grain_note="A different question from safety stock. The two thresholds "
                   "are configured independently and neither list contains the "
                   "other by construction.",
        run=analytics.products_needing_reorder,
    ),
    "pending_purchase_order_count": SqlTool(
        name="pending_purchase_order_count",
        title="Pending purchase order count",
        description="How many purchase orders are pending, counted at order grain.",
        grain="order",
        grain_note="Purchase ORDERS, not order lines: the same pending orders "
                   "have more detail lines than headers.",
        run=_pending_purchase_order_count,
    ),
    "pending_purchase_order_value": SqlTool(
        name="pending_purchase_order_value",
        title="Value of pending purchase orders",
        description="Committed merchandise and total value of the pending orders, "
                    "at order grain.",
        grain="order",
        grain_note="Header-level commitment. Line-level purchase amounts are a "
                   "separate measure and are not summed into this.",
        run=analytics.pending_purchase_order_value,
    ),
    "monthly_sales_trends": SqlTool(
        name="monthly_sales_trends",
        title="Monthly sales trends",
        description="Per calendar month: orders, order-level merchandise and "
                    "total due, lines, line-level revenue and the month-on-month "
                    "change in line revenue.",
        grain="order and line, aggregated separately then joined on the month",
        grain_note="Both grains per month, each aggregated in its own branch. "
                   "The month bucket is a date, not a timestamp.",
        run=analytics.monthly_sales_trends,
    ),
}


def tool_names() -> List[str]:
    """Names in the order the Phase 2 brief lists the questions."""
    return [name for name in analytics.QUESTIONS if name in ALLOWLIST]


def get(name: str) -> SqlTool:
    tool = ALLOWLIST.get(name)
    if tool is None:
        raise UnknownQuery(f"{name!r} is not an approved query")
    return tool


def catalogue_titles() -> List[str]:
    """Human titles in brief order, for telling a caller what is answerable.

    An "unsupported" answer is only useful if it names the alternatives, and
    naming them with titles rather than query identifiers keeps the suggestion
    readable to someone who has never seen the allowlist.
    """
    return [ALLOWLIST[name].title for name in tool_names()]


def catalog() -> str:
    """The tool list the classifier is shown: names, titles and descriptions.

    Deliberately contains no SQL, so a model cannot copy syntax back and
    cannot be influenced by table or column names it might invent around.
    """
    lines = []
    for name in tool_names():
        tool = ALLOWLIST[name]
        parameters = ", ".join(tool.params) if tool.params else "none"
        lines.append(f"- {tool.name}: {tool.title}. {tool.description} "
                     f"[parameters: {parameters}]")
    return "\n".join(lines)


# --------------------------------------------------------------- normalisation

def cell(value: Any) -> Any:
    """A JSON-safe cell that keeps every stored digit.

    Decimal becomes str rather than float: USD 123216786.1159 printed as
    123216786.11589998 would be the first step back to the rounding problems
    Phase 2 removed.
    """
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value


def _as_rows(result: Any) -> List[Dict[str, Any]]:
    """Any of the three shapes analytics returns, as a list of dict rows.

    A dict is one row, a list is already rows, and a bare number would have no
    column name, which is why no tool is allowed to return one (see
    _pending_purchase_order_count).
    """
    if isinstance(result, dict):
        return [{key: cell(value) for key, value in result.items()}]
    if isinstance(result, list):
        return [{key: cell(value) for key, value in row.items()}
                for row in result]
    raise TypeError(f"approved query returned {type(result).__name__}, "
                    f"which has no column name to report it under")


@dataclass(frozen=True)
class QueryResult:
    name: str
    title: str
    grain: str
    grain_note: str
    parameters: Dict[str, Any]
    columns: List[str] = field(default_factory=list)
    rows: List[Dict[str, Any]] = field(default_factory=list)
    truncated: bool = False

    @property
    def row_count(self) -> int:
        return len(self.rows)

    def to_dict(self, max_rows: Optional[int] = None) -> Dict[str, Any]:
        """Response body for the API. `max_rows` bounds the payload, not the
        answer: truncation is declared, never silent."""
        rows = self.rows if max_rows is None else self.rows[:max_rows]
        return {
            "query": self.name,
            "title": self.title,
            "grain": self.grain,
            "grain_note": self.grain_note,
            "parameters": self.parameters,
            "columns": self.columns,
            "row_count": self.row_count,
            "returned_rows": len(rows),
            "truncated": max_rows is not None and self.row_count > len(rows),
            "rows": rows,
        }

    def render(self, max_rows: int) -> str:
        """The text block a language model reads, or a human sees with no model.

        Every figure arrives exactly as the database returned it, labelled with
        its grain, and an empty result says so instead of leaving room for an
        invented number.
        """
        if not self.rows:
            return (f"{self.title}: the query ran successfully and returned no "
                    f"rows. There is no figure to report.")
        shown = self.rows[:max_rows]
        lines = [f"{self.title} [{self.grain} grain] - {self.row_count} "
                 f"rows returned"]
        for row in shown:
            lines.append("  " + " ".join(f"{column}={value}"
                                         for column, value in row.items()))
        if len(self.rows) > len(shown):
            # The count is stated above so a reader of a truncated block cannot
            # mistake "15 rows shown" for "15 rows exist".
            lines.append(f"  ... {len(self.rows) - len(shown)} more rows not "
                         f"shown; the row count above is the complete result")
        lines.append(f"note: {self.grain_note}")
        return "\n".join(lines)


def execute(conn, name: str, parameters: Optional[Dict[str, Any]] = None,
            limit: int = analytics.DEFAULT_LIMIT,
            limit_ceiling: int = 50) -> QueryResult:
    """Run one approved query by name.

    `limit` is the only value that reaches SQL as a parameter, it is coerced to
    an integer and clamped here, and the analytics module binds it with a
    placeholder rather than interpolating it.
    """
    tool = get(name)
    supplied = dict(parameters or {})
    unknown = set(supplied) - set(tool.params)
    if unknown:
        raise ValueError(f"{name} does not accept {sorted(unknown)}")

    arguments: Dict[str, Any] = {}
    if "limit" in tool.params:
        arguments["limit"] = clamp_limit(supplied.get("limit", limit),
                                         limit_ceiling)

    result = tool.run(conn, **arguments) if arguments else tool.run(conn)
    rows = _as_rows(result)
    columns = list(rows[0]) if rows else []
    return QueryResult(name=tool.name, title=tool.title, grain=tool.grain,
                       grain_note=tool.grain_note,
                       parameters=dict(arguments), columns=columns, rows=rows)


def clamp_limit(value: Any, ceiling: int) -> int:
    """A positive integer no larger than the ceiling.

    Accepts the string a JSON payload may carry and refuses to guess at
    anything that is not a number, rather than silently reading "10; DROP"
    as 10.
    """
    if isinstance(value, bool):
        raise ValueError("limit must be an integer, not a boolean")
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"limit must be a whole number, got {value!r}")
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"limit must be an integer, got {value!r}")
    if number <= 0:
        raise ValueError(f"limit must be greater than zero, got {number}")
    return min(number, ceiling)
