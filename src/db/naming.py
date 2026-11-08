"""Naming bridge between the raw AdventureWorks files and the database.

The staging schema is a mirror of the files, and `loader.HEADERS` already is
the authoritative description of those files. Deriving the staging identifiers
from it, instead of typing them out a second time, means `db/staging.sql` can
be checked against a single source of truth: if a column is renamed in the
loader, the staging column assertion in tests/test_db_postgres.py fails rather
than silently shifting every value by one position.

Two naming worlds exist deliberately:

* staging   = mechanical snake_case of the source column, so a mirror can be
              compared to the file row by row (SubTotal -> sub_total).
* erp       = business names chosen for readability in queries and comments
              (SubTotal -> sub_total, OnlineOrderFlag -> online_order,
              ModifiedDate -> modified_at, Class -> product_class).

The mapping between them is written out in db/transform.sql, not computed, so
the renames stay reviewable.
"""
import re
from typing import Dict, List

from src.data import loader

# "SalesOrderID" -> "sales_order_id", "PurchaseOrderDetailID" ->
# "purchase_order_detail_id", "rowguid" -> "rowguid".
_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def to_snake(name: str) -> str:
    """Convert one AdventureWorks PascalCase identifier to snake_case."""
    return _BOUNDARY.sub("_", name).lower()


def staging_table(raw_name: str) -> str:
    """stg table mirroring the raw file `<raw_name>.csv`."""
    return f"stg.{to_snake(raw_name)}"


def staging_columns(raw_name: str) -> List[str]:
    """Column names of the staging mirror, in file order."""
    return [to_snake(column) for column in loader.HEADERS[raw_name]]


# Raw file -> erp table. The erp names describe the business object, which is
# why SalesOrderHeader becomes sales_order and not sales_order_header.
ERP_TABLES: Dict[str, str] = {
    "SalesOrderHeader":    "sales_order",
    "SalesOrderDetail":    "sales_order_line",
    "Customer":            "customer",
    "Product":             "product",
    "ProductInventory":    "product_inventory",
    "PurchaseOrderHeader": "purchase_order",
    "PurchaseOrderDetail": "purchase_order_line",
}

# Insertion order matters: parents before children, or the foreign keys reject
# the load. This is the same order db/transform.sql is written in.
ERP_LOAD_ORDER = [
    "product",
    "customer",
    "sales_order",
    "sales_order_line",
    "product_inventory",
    "purchase_order",
    "purchase_order_line",
]
