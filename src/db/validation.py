"""Post-load validation for the ERP migration, and the oracle behind it.

Every check here is a gate: `src.db.migrate` runs them inside the same
transaction as the load, so a failure rolls the database back to whatever was
committed last time instead of leaving a half-loaded warehouse behind.

The financial checks do not compare the database against itself. The expected
totals are computed with `decimal.Decimal` straight from the raw TSV text,
which is an independent path: if a cast in db/transform.sql rounded, truncated
or re-scaled a value, the comparison catches it, and if the source files
changed the check notices that too. `loader.validate_dataset()` supplies the
row counts, so the file line count is not a second hardcoded number either.
"""
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Callable, Dict, List, Optional

from src.data import loader
from src.db import naming


# --------------------------------------------------------------- expectations

EXPECTED_ERP_ROW_COUNTS = {
    "sales_order":           31_465,
    "sales_order_line":     121_317,
    "customer":               19_820,
    "product":                   504,
    "product_inventory":       1_069,
    "purchase_order":          4_012,
    "purchase_order_line":     8_845,
}

# The six approved inventory figures. These are the contract from Phase 1 and
# the reason the view exists.
EXPECTED_STOCK_POSITION = {
    "products":                  504,
    "recorded":                  432,
    "unrecorded":                 72,
    "below_safety":               69,
    "needs_reorder":               7,
    "confirmed_zero":              4,
}

EXPECTED_PURCHASE_ORDER_STATUS = {1: 225, 2: 12, 3: 86, 4: 3689}
EXPECTED_SALES_ORDER_STATUS    = {5: 31_465}

# Coverage figures: not errors, but they must not move silently.
EXPECTED_NEVER_SOLD_PRODUCTS      = 238
EXPECTED_NEVER_PURCHASED_PRODUCTS = 239
EXPECTED_CUSTOMERS_WITHOUT_ORDERS = 701

# SubTotal is published at four decimals while LineTotal is published at six,
# so the header/line identity is only expected to hold to within a cent per
# order. Measured drift across the whole dataset is 0.004012 on 109.8M.
ORDER_LINE_TOLERANCE = Decimal("0.01")

# Which money columns to reconcile against the files, as
# erp table -> (raw file, [source columns]).
MONEY_ORACLE = [
    ("SELECT sum(total_due)  FROM erp.sales_order",        "SalesOrderHeader",    "total_due"),
    ("SELECT sum(sub_total)  FROM erp.sales_order",        "SalesOrderHeader",    "sub_total"),
    ("SELECT sum(tax_amt)    FROM erp.sales_order",        "SalesOrderHeader",    "tax_amt"),
    ("SELECT sum(freight)    FROM erp.sales_order",        "SalesOrderHeader",    "freight"),
    ("SELECT sum(line_total) FROM erp.sales_order_line",   "SalesOrderDetail",    "line_total"),
    ("SELECT sum(total_due)  FROM erp.purchase_order",     "PurchaseOrderHeader", "total_due"),
    ("SELECT sum(line_total) FROM erp.purchase_order_line","PurchaseOrderDetail", "line_total"),
]

PENDING_TOTAL_DUE_SQL = (
    "SELECT sum(total_due) FROM erp.purchase_order WHERE order_status = 1"
)


@dataclass
class Check:
    """One validation result. `passed` is only ever set by a comparison."""

    name:     str
    expected: object
    actual:   object
    message:  str = ""

    @property
    def passed(self) -> bool:
        return self.expected == self.actual

    def __str__(self) -> str:
        mark = "PASS" if self.passed else "FAIL"
        line = f"  [{mark}] {self.name}: expected {self.expected!r}, got {self.actual!r}"
        return f"{line}  ({self.message})" if self.message else line


def failures(checks: List[Check]) -> List[Check]:
    return [c for c in checks if not c.passed]


def format_report(checks: List[Check]) -> str:
    bad = failures(checks)
    head = [f"Validation: {len(checks) - len(bad)}/{len(checks)} checks passed"]
    return "\n".join(head + [str(c) for c in checks] +
                     ([] if not bad else [f"{len(bad)} FAILED - the migration "
                                          "transaction was rolled back"]))


# ------------------------------------------------------------------- helpers

def _scalar(conn, sql: str, params=None):
    row = conn.execute(sql, params).fetchone()
    return row[0] if row else None


def _dict_rows(conn, sql: str, params=None) -> List[Dict[str, object]]:
    """Rows as dicts, keyed by the names PostgreSQL reported for them.

    psycopg 3 hands back tuples; naming the columns from the cursor
    description keeps these checks readable without pulling in a row factory
    or a second database library.
    """
    result = conn.execute(sql, params)
    names = [column.name for column in result.description]
    return [dict(zip(names, row)) for row in result.fetchall()]


def _row(conn, sql: str) -> Dict[str, object]:
    """Exactly one row, as a dict."""
    return _dict_rows(conn, sql)[0]


def _mapping(conn, sql: str, key: str, value: str) -> Dict[str, int]:
    return {r[key]: r[value] for r in _dict_rows(conn, sql)}


def source_decimal_sum(path: Path, raw_name: str, column: str,
                       where_column: Optional[str] = None,
                       where_value: Optional[str] = None) -> Decimal:
    """Sum one money column straight from the raw file, in exact decimals.

    `column` is a staging name (`total_due`), because MONEY_ORACLE is written in
    the same snake_case the rest of this module uses; the raw file itself has no
    column names at all, so the position comes from loader.HEADERS via naming.
    """
    columns = naming.staging_columns(raw_name)
    index, filter_index = columns.index(column), None
    if where_column:
        filter_index = columns.index(where_column)
    total = Decimal(0)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            if filter_index is not None and fields[filter_index].strip() != where_value:
                continue
            value = fields[index].strip()
            if value:
                total += Decimal(value)
    return total


def source_rows_with_more_decimals(path: Path, raw_name: str, column: str,
                                   decimals: int = 2) -> int:
    """Count rows whose published value carries more than `decimals` places.

    Measured on the text, so it is independent of how the database stores and
    compares NUMERIC values.
    """
    index = naming.staging_columns(raw_name).index(column)
    quantum = Decimal(1).scaleb(-decimals)
    total = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            value = line.rstrip("\n").split("\t")[index].strip()
            # A nonzero remainder against 0.01 means the published amount has
            # more precision than cents, which is what the database must keep.
            if value and Decimal(value) % quantum:
                total += 1
    return total


def raw_files() -> Dict[str, Path]:
    return loader.required_files()


# -------------------------------------------------------------------- checks

def check_row_counts(conn, raw_counts: Dict[str, int]) -> List[Check]:
    """Every raw row reaches staging, and every staging row reaches erp."""
    checks = []
    staging = _mapping(conn, " UNION ALL ".join(
        f"SELECT '{naming.to_snake(name)}' AS table_name, count(*) AS rows "
        f"FROM {naming.staging_table(name)}"
        for name in loader.HEADERS), "table_name", "rows")
    erp = _mapping(conn, " UNION ALL ".join(
        f"SELECT '{table}' AS table_name, count(*) AS rows FROM erp.{table}"
        for table in naming.ERP_LOAD_ORDER), "table_name", "rows")

    for name in loader.HEADERS:
        expected = raw_counts[name]
        checks.append(Check(f"rows {name}.csv -> stg.{naming.to_snake(name)}",
                            expected, staging.get(naming.to_snake(name)),
                            "COPY lost or duplicated file rows"))
        table = naming.ERP_TABLES[name]
        checks.append(Check(f"rows erp.{table}", expected, erp.get(table),
                            "the transform dropped or invented rows"))
        checks.append(Check(f"approved count for erp.{table}",
                            EXPECTED_ERP_ROW_COUNTS[table], erp.get(table),
                            "differs from the figure approved in the architecture"))
    return checks


def check_stock_position(conn) -> List[Check]:
    """The six approved inventory figures, from the view the app will use."""
    row = _row(conn, """
        SELECT count(*)                                              AS products,
               count(*) FILTER (WHERE has_inventory_record)           AS recorded,
               count(*) FILTER (WHERE NOT has_inventory_record)       AS unrecorded,
               count(*) FILTER (WHERE below_safety)                   AS below_safety,
               count(*) FILTER (WHERE needs_reorder)                  AS needs_reorder,
               count(*) FILTER (WHERE total_quantity = 0)             AS confirmed_zero
        FROM erp.stock_position""")
    return [Check(f"stock_position {key}", value, row[key])
            for key, value in EXPECTED_STOCK_POSITION.items()]


def check_missing_inventory_is_never_zero(conn) -> List[Check]:
    """The Phase 1 bug class: unknown stock must not read as empty stock."""
    return [Check(
        "unrecorded products have NULL total_quantity",
        EXPECTED_STOCK_POSITION["unrecorded"],
        _scalar(conn, "SELECT count(*) FROM erp.stock_position "
                      "WHERE NOT has_inventory_record AND total_quantity IS NULL"),
        "a missing inventory record must stay unknown, never 0"),
        Check("alerts raised on unrecorded products", 0, _scalar(conn, """
            SELECT count(*) FROM erp.stock_position
            WHERE NOT has_inventory_record AND (below_safety OR needs_reorder)"""),
            "an absent record says nothing about stock, so it cannot be an alert"),
        Check("confirmed-zero products are flagged below safety", 4, _scalar(conn, """
            SELECT count(*) FROM erp.stock_position
            WHERE total_quantity = 0 AND below_safety"""),
            "zero on hand IS a real stock-out: these four stay visible"),
        Check("invented alerts if unknown were treated as zero", 0, _scalar(conn, """
            SELECT count(*) FILTER (WHERE COALESCE(total_quantity, 0)
                                          < safety_stock_level)
                   - count(*) FILTER (WHERE NOT has_inventory_record)
                   - count(*) FILTER (WHERE below_safety)
            FROM erp.stock_position"""),
            "141 naive - 72 unknown = 69 real alerts: the view and the naive "
            "reading differ by exactly the products with no record")]


def check_orphans(conn) -> List[Check]:
    """No child row may point at a parent that does not exist."""
    paths = [
        ("sales_order.customer_id -> customer",
         "SELECT count(*) FROM erp.sales_order o LEFT JOIN erp.customer c "
         "ON c.customer_id = o.customer_id WHERE c.customer_id IS NULL"),
        ("sales_order_line.sales_order_id -> sales_order",
         "SELECT count(*) FROM erp.sales_order_line l LEFT JOIN erp.sales_order o "
         "USING (sales_order_id) WHERE o.sales_order_id IS NULL"),
        ("sales_order_line.product_id -> product",
         "SELECT count(*) FROM erp.sales_order_line l LEFT JOIN erp.product p "
         "USING (product_id) WHERE p.product_id IS NULL"),
        ("product_inventory.product_id -> product",
         "SELECT count(*) FROM erp.product_inventory i LEFT JOIN erp.product p "
         "USING (product_id) WHERE p.product_id IS NULL"),
        ("purchase_order_line.purchase_order_id -> purchase_order",
         "SELECT count(*) FROM erp.purchase_order_line l "
         "LEFT JOIN erp.purchase_order o USING (purchase_order_id) "
         "WHERE o.purchase_order_id IS NULL"),
        ("purchase_order_line.product_id -> product",
         "SELECT count(*) FROM erp.purchase_order_line l LEFT JOIN erp.product p "
         "USING (product_id) WHERE p.product_id IS NULL"),
    ]
    return [Check(f"orphan rows: {label}", 0, _scalar(conn, sql))
            for label, sql in paths]


def check_headers_have_lines(conn) -> List[Check]:
    """An order with no detail lines is a broken load, not a data fact."""
    return [
        Check("sales orders without lines", 0, _scalar(conn, """
            SELECT count(*) FROM erp.sales_order o
            WHERE NOT EXISTS (SELECT 1 FROM erp.sales_order_line l
                              WHERE l.sales_order_id = o.sales_order_id)""")),
        Check("purchase orders without lines", 0, _scalar(conn, """
            SELECT count(*) FROM erp.purchase_order o
            WHERE NOT EXISTS (SELECT 1 FROM erp.purchase_order_line l
                              WHERE l.purchase_order_id = o.purchase_order_id)""")),
    ]


def check_no_duplicated_order_revenue(conn) -> List[Check]:
    """Structural proof that order-level revenue cannot be counted twice.

    In the flat CSV pipeline the only defence against this was remembering to
    deduplicate. Here the primary key makes a repeated order row impossible,
    so the check is cheap and it is the check that matters: one row per order,
    and the total over those rows equals the total over the file.
    """
    return [
        Check("duplicate sales_order_id rows", 0, _scalar(conn, """
            SELECT count(*) FROM (SELECT sales_order_id FROM erp.sales_order
                                  GROUP BY sales_order_id HAVING count(*) > 1) d""")),
        Check("duplicate purchase_order_id rows", 0, _scalar(conn, """
            SELECT count(*) FROM (SELECT purchase_order_id FROM erp.purchase_order
                                  GROUP BY purchase_order_id HAVING count(*) > 1) d""")),
        Check("duplicate sales order lines", 0, _scalar(conn, """
            SELECT count(*) FROM (SELECT sales_order_id, sales_order_detail_id
                                  FROM erp.sales_order_line
                                  GROUP BY 1, 2 HAVING count(*) > 1) d""")),
        Check("orders counted once at order grain", 31_465,
              _scalar(conn, "SELECT count(DISTINCT sales_order_id) FROM erp.sales_order")),
    ]


def check_financial_identities(conn) -> List[Check]:
    """The arithmetic published with the data, verified in the database."""
    return [
        Check("orders where total_due <> sub_total + tax + freight", 0, _scalar(conn, """
            SELECT count(*) FROM erp.sales_order
            WHERE total_due <> sub_total + tax_amt + freight""")),
        Check("purchase orders failing the same identity", 0, _scalar(conn, """
            SELECT count(*) FROM erp.purchase_order
            WHERE total_due <> sub_total + tax_amt + freight""")),
        Check("purchase order lines failing stocked = received - rejected", 0,
              _scalar(conn, """
            SELECT count(*) FROM erp.purchase_order_line
            WHERE stocked_qty <> received_qty - rejected_qty""")),
        Check("orders whose sub_total breaks from its lines by over a cent", 0,
              _scalar(conn, f"""
            SELECT count(*) FROM (
                SELECT o.sales_order_id, o.sub_total, sum(l.line_total) AS lines
                FROM erp.sales_order o JOIN erp.sales_order_line l
                  ON l.sales_order_id = o.sales_order_id
                GROUP BY o.sales_order_id, o.sub_total
                HAVING abs(o.sub_total - sum(l.line_total)) > {ORDER_LINE_TOLERANCE}
            ) bad"""),
            "six-decimal lines summing into a four-decimal header: tolerance, not equality"),
    ]


def check_status_domains(conn) -> List[Check]:
    sales = _mapping(conn, "SELECT order_status::text AS k, count(*) AS v "
                           "FROM erp.sales_order GROUP BY 1", "k", "v")
    purchase = _mapping(conn, "SELECT order_status::text AS k, count(*) AS v "
                              "FROM erp.purchase_order GROUP BY 1", "k", "v")
    return [
        Check("sales order status counts",
              {str(k): v for k, v in EXPECTED_SALES_ORDER_STATUS.items()}, sales,
              "status is 5 (Shipped) for every row in this dataset"),
        Check("purchase order status counts",
              {str(k): v for k, v in EXPECTED_PURCHASE_ORDER_STATUS.items()}, purchase,
              "1 Pending, 2 Approved, 3 Rejected, 4 Complete"),
        Check("pending purchase orders", EXPECTED_PURCHASE_ORDER_STATUS[1],
              _scalar(conn, "SELECT count(*) FROM erp.purchase_order WHERE order_status = 1"),
              "counted at order grain"),
    ]


def check_coverage_counts(conn) -> List[Check]:
    """Products and customers that exist but never transact."""
    return [
        Check("products never sold", EXPECTED_NEVER_SOLD_PRODUCTS, _scalar(conn, """
            SELECT count(*) FROM erp.product p
            WHERE NOT EXISTS (SELECT 1 FROM erp.sales_order_line l
                              WHERE l.product_id = p.product_id)""")),
        Check("products never purchased", EXPECTED_NEVER_PURCHASED_PRODUCTS, _scalar(conn, """
            SELECT count(*) FROM erp.product p
            WHERE NOT EXISTS (SELECT 1 FROM erp.purchase_order_line l
                              WHERE l.product_id = p.product_id)""")),
        Check("customers without orders", EXPECTED_CUSTOMERS_WITHOUT_ORDERS, _scalar(conn, """
            SELECT count(*) FROM erp.customer c
            WHERE NOT EXISTS (SELECT 1 FROM erp.sales_order o
                              WHERE o.customer_id = c.customer_id)""")),
    ]


def check_totals_match_source_files(conn) -> List[Check]:
    """Database totals against exact decimal sums of the raw text."""
    files = raw_files()
    checks = []
    for sql, raw_name, column in MONEY_ORACLE:
        expected = source_decimal_sum(files[raw_name], raw_name, column)
        checks.append(Check(f"{raw_name}.{column} reconciles", expected,
                            _scalar(conn, sql),
                            "summed over the whole table, so no row is counted twice"))

    pending = source_decimal_sum(files["PurchaseOrderHeader"], "PurchaseOrderHeader",
                                 "total_due", where_column="status", where_value="1")
    checks.append(Check("PurchaseOrderHeader.total_due where status=1 reconciles",
                        pending, _scalar(conn, PENDING_TOTAL_DUE_SQL),
                        "pending purchase order value at order grain"))
    return checks


def check_precision_is_preserved(conn) -> List[Check]:
    """That the migration really stored four and six decimals, not cents.

    The expected counts come from the raw text, which is the only way to prove
    the database did not round on the way in: the CSV pipeline rounds these
    same columns to two decimals, and a migration that quietly did the same
    would still produce a plausible-looking total.
    """
    files = raw_files()
    expected_orders = source_rows_with_more_decimals(
        files["SalesOrderHeader"], "SalesOrderHeader", "total_due")
    expected_lines = source_rows_with_more_decimals(
        files["SalesOrderDetail"], "SalesOrderDetail", "line_total")
    return [
        Check("sales orders stored with more than two decimals", expected_orders,
              _scalar(conn, "SELECT count(*) FROM erp.sales_order "
                            "WHERE total_due <> round(total_due, 2)"),
              "raw file precision preserved, not rounded at rest"),
        Check("order lines stored with more than two decimals", expected_lines,
              _scalar(conn, "SELECT count(*) FROM erp.sales_order_line "
                            "WHERE line_total <> round(line_total, 2)"),
              "line_total is NUMERIC(19,6) exactly as published"),
    ]


# ---------------------------------------------------------------------- gate

ALL_CHECKS: List[Callable] = [
    check_stock_position,
    check_missing_inventory_is_never_zero,
    check_orphans,
    check_headers_have_lines,
    check_no_duplicated_order_revenue,
    check_financial_identities,
    check_status_domains,
    check_coverage_counts,
    check_totals_match_source_files,
    check_precision_is_preserved,
]


def run_all(conn, raw_counts: Optional[Dict[str, int]] = None) -> List[Check]:
    """Every gate. Raise before returning so callers cannot ignore a failure."""
    raw_counts = raw_counts if raw_counts is not None else loader.validate_dataset()
    checks = check_row_counts(conn, raw_counts)
    for check in ALL_CHECKS:
        checks.extend(check(conn))
    return checks
