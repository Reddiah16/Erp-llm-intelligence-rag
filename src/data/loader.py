"""AdventureWorks loader for the canonical oltp-install-script dataset.

The canonical files are the Microsoft sql-server-samples
``oltp-install-script/*.csv`` exports, which are TAB-delimited, headerless
and UTF-8/ANSI encoded. Column names come from HEADERS below, so the loader
never trusts a header row. See the "Required dataset" section in the README.
"""
import codecs
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")
log = logging.getLogger(__name__)

# Paths are anchored to the repository root so the loader behaves identically
# no matter which directory the terminal was started in. No OS-specific paths.
ROOT_DIR = Path(__file__).resolve().parents[2]
SD_DIR = ROOT_DIR / "data" / "sd"
MM_DIR = ROOT_DIR / "data" / "mm"

EXPECTED_FORMAT  = "tab-delimited (\\t), headerless, UTF-8/ANSI encoded"
DATASET_SOURCE   = ("https://github.com/Microsoft/sql-server-samples/"
                    "tree/master/samples/databases/adventure-works/oltp-install-script")

# Published AdventureWorks row counts, used only to warn about truncated
# downloads — never treated as a hard failure, other versions are legitimate.
REFERENCE_ROW_COUNTS = {
    "SalesOrderHeader":    31465,
    "SalesOrderDetail":   121317,
    "Customer":             19820,
    "Product":                504,
    "ProductInventory":      1069,
    "PurchaseOrderHeader":   4012,
    "PurchaseOrderDetail":   8845,
}
ROW_COUNT_TOLERANCE = 0.9

HEADERS = {
    "SalesOrderHeader": [
        "SalesOrderID","RevisionNumber","OrderDate","DueDate","ShipDate",
        "Status","OnlineOrderFlag","SalesOrderNumber","PurchaseOrderNumber",
        "AccountNumber","CustomerID","SalesPersonID","TerritoryID",
        "BillToAddressID","ShipToAddressID","ShipMethodID","CreditCardID",
        "CreditCardApprovalCode","CurrencyRateID","SubTotal","TaxAmt",
        "Freight","TotalDue","Comment","rowguid","ModifiedDate"
    ],
    "SalesOrderDetail": [
        "SalesOrderID","SalesOrderDetailID","CarrierTrackingNumber",
        "OrderQty","ProductID","SpecialOfferID","UnitPrice",
        "UnitPriceDiscount","LineTotal","rowguid","ModifiedDate"
    ],
    "Customer": [
        "CustomerID","PersonID","StoreID","TerritoryID",
        "AccountNumber","rowguid","ModifiedDate"
    ],
    "Product": [
        "ProductID","Name","ProductNumber","MakeFlag","FinishedGoodsFlag",
        "Color","SafetyStockLevel","ReorderPoint","StandardCost","ListPrice",
        "Size","SizeUnitMeasureCode","WeightUnitMeasureCode","Weight",
        "DaysToManufacture","ProductLine","Class","Style",
        "ProductSubcategoryID","ProductModelID","SellStartDate",
        "SellEndDate","DiscontinuedDate","rowguid","ModifiedDate"
    ],
    "ProductInventory": [
        "ProductID","LocationID","Shelf","Bin","Quantity",
        "rowguid","ModifiedDate"
    ],
    "PurchaseOrderHeader": [
        "PurchaseOrderID","RevisionNumber","Status","EmployeeID",
        "VendorID","ShipMethodID","OrderDate","ShipDate",
        "SubTotal","TaxAmt","Freight","TotalDue","ModifiedDate"
    ],
    "PurchaseOrderDetail": [
        "PurchaseOrderID","PurchaseOrderDetailID","DueDate","OrderQty",
        "ProductID","UnitPrice","LineTotal","ReceivedQty",
        "RejectedQty","StockedQty","ModifiedDate"
    ],
}

DROP_COLS = ["rowguid", "ModifiedDate", "Comment", "CreditCardApprovalCode",
             "CurrencyRateID", "RevisionNumber"]


class DatasetError(RuntimeError):
    """The AdventureWorks dataset is missing, incomplete or malformed."""


def required_files() -> Dict[str, Path]:
    """The seven raw files the loader needs, as {HEADERS key: path}.

    Resolved at call time so an alternate data location can be selected by
    rebinding SD_DIR / MM_DIR (used by the test fixtures).
    """
    return {
        "SalesOrderHeader":    SD_DIR / "SalesOrderHeader.csv",
        "SalesOrderDetail":    SD_DIR / "SalesOrderDetail.csv",
        "Customer":            SD_DIR / "Customer.csv",
        "Product":             MM_DIR / "Product.csv",
        "ProductInventory":    MM_DIR / "ProductInventory.csv",
        "PurchaseOrderHeader": MM_DIR / "PurchaseOrderHeader.csv",
        "PurchaseOrderDetail": MM_DIR / "PurchaseOrderDetail.csv",
    }


def _inspect_file(name: str, path: Path, sample_lines: int = 20,
                  count_rows: bool = True) -> Tuple[List[str], int]:
    """Inspect one raw file without loading it.

    Returns (problems, row_count); problems is empty when the file is usable.
    """
    expected_cols = len(HEADERS[name])

    if not path.exists():
        return [f"{name}: file not found — expected at {path}"], 0
    if path.stat().st_size == 0:
        return [f"{name}: file is empty — {path}"], 0

    with path.open("rb") as fh:
        head = fh.read(4096)
    if head.startswith(codecs.BOM_UTF8):
        return [f"{name}: {path.name} starts with a UTF-8 BOM; expected {EXPECTED_FORMAT}"], 0
    if (head.startswith(codecs.BOM_UTF16_LE) or head.startswith(codecs.BOM_UTF16_BE)
            or b"\x00" in head[:32]):
        return [f"{name}: {path.name} looks UTF-16 encoded; expected {EXPECTED_FORMAT}"], 0
    try:
        # final=False buffers an incomplete multi-byte character that happens
        # to straddle the probe boundary, instead of reporting a false fault.
        codecs.getincrementaldecoder("utf-8")().decode(head, final=False)
    except UnicodeDecodeError as exc:
        return [f"{name}: {path.name} is not valid UTF-8/ANSI ({exc}); expected {EXPECTED_FORMAT}"], 0

    header_tokens = {h.lower() for h in HEADERS[name]}
    counts: Dict[int, int] = {}
    rows = 0
    with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        for line in fh:
            rows += 1
            if rows <= sample_lines:
                fields = [f.strip() for f in line.rstrip("\r\n").split("\t")]
                counts[len(fields)] = counts.get(len(fields), 0) + 1
                if rows == 1:
                    if set(f.lower() for f in fields) & header_tokens:
                        return [f"{name}: {path.name} appears to carry a header row (first line "
                               f"contains column names). The canonical files are headerless — the "
                               f"loader supplies its own {expected_cols} column names."], rows
                    if len(fields) == 1 and "," in line:
                        return [f"{name}: {path.name} appears comma-delimited, not tab-delimited. "
                               f"Expected {EXPECTED_FORMAT}; re-download from {DATASET_SOURCE}."], rows
                    if len(fields) == 1:
                        return [f"{name}: no tab separator found in {path.name}. "
                               f"Expected {EXPECTED_FORMAT}."], rows
            elif not count_rows:
                break

    problems: List[str] = []
    if rows == 0:
        problems.append(f"{name}: {path.name} contains no data rows")
    elif len(counts) > 1:
        problems.append(f"{name}: {path.name} has inconsistent column counts "
                       f"{sorted(counts)} within the first {sample_lines} rows; expected {expected_cols}")
    elif counts and next(iter(counts)) != expected_cols:
        problems.append(f"{name}: {path.name} has {next(iter(counts))} tab-separated columns, "
                       f"expected {expected_cols} (HEADERS['{name}'])")
    return problems, rows


def validate_dataset(sample_lines: int = 20, raise_on_error: bool = True) -> Dict[str, int]:
    """Preflight the whole dataset before any loading, chunking or embedding.

    Returns {table: row_count}. Raises DatasetError listing every failing file
    with its expected format, so a partial or foreign-format dataset is never
    ingested into the vector store.
    """
    problems: List[str] = []
    table_rows: Dict[str, int] = {}

    for name, path in required_files().items():
        file_problems, rows = _inspect_file(name, path, sample_lines=sample_lines)
        table_rows[name] = rows
        if file_problems:
            problems.append(file_problems[0])
            continue
        reference = REFERENCE_ROW_COUNTS.get(name)
        if reference and rows < int(reference * ROW_COUNT_TOLERANCE):
            log.warning(f"{name}: only {rows:,} rows found, AdventureWorks publishes "
                       f"~{reference:,} — possible truncated download")
        log.info(f"  OK {path.name}: {rows:,} rows x {len(HEADERS[name])} columns")

    if problems and raise_on_error:
        raise DatasetError(
            "AdventureWorks dataset preflight failed:\n  - " + "\n  - ".join(problems) +
            f"\n\nExpected format per file: {EXPECTED_FORMAT}"
            f"\nExpected location: {SD_DIR} and {MM_DIR}"
            f"\nSource: {DATASET_SOURCE}"
        )
    return table_rows


def read_csv(path: Path, name: str) -> pd.DataFrame:
    problems, _ = _inspect_file(name, path, sample_lines=5, count_rows=False)
    if problems:
        raise DatasetError(problems[0])

    df = pd.read_csv(
        path, sep="\t", header=None,
        names=HEADERS[name], low_memory=False
    )
    return df


def clean_dates(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if "Date" in col or "date" in col:
            try:
                df[col] = pd.to_datetime(df[col], errors="coerce").dt.strftime("%Y-%m-%d")
            except Exception:
                pass
    return df


def drop_unused(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in DROP_COLS if c in df.columns]
    return df.drop(columns=cols)


def load_sales_orders() -> pd.DataFrame:
    log.info("Loading sales orders...")
    header = read_csv(SD_DIR / "SalesOrderHeader.csv", "SalesOrderHeader")
    detail = read_csv(SD_DIR / "SalesOrderDetail.csv", "SalesOrderDetail")
    product = read_csv(MM_DIR / "Product.csv", "Product")[
        ["ProductID", "Name", "ProductNumber"]
    ]

    # Join detail with product names
    detail = detail.merge(product, on="ProductID", how="left")

    # Join header with enriched detail
    df = header.merge(
        detail[["SalesOrderID", "ProductID", "Name",
                "ProductNumber", "OrderQty", "UnitPrice", "LineTotal"]],
        on="SalesOrderID", how="left"
    )

    df = clean_dates(df)
    df = drop_unused(df)
    df["TotalDue"] = df["TotalDue"].round(2)
    df["Status"]   = df["Status"].map({
        1: "In Process", 2: "Approved", 3: "Backordered",
        4: "Rejected",   5: "Shipped",  6: "Cancelled"
    }).fillna("Unknown")

    log.info(f"Sales orders: {len(df)} rows")
    return df


def load_customers() -> pd.DataFrame:
    log.info("Loading customers...")
    df = read_csv(SD_DIR / "Customer.csv", "Customer")
    df = clean_dates(df)
    df = drop_unused(df)
    log.info(f"Customers: {len(df)} rows")
    return df


def load_products() -> pd.DataFrame:
    log.info("Loading products...")
    df = read_csv(MM_DIR / "Product.csv", "Product")
    df = clean_dates(df)
    df = drop_unused(df)
    df["ListPrice"]     = df["ListPrice"].fillna(0).round(2)
    df["StandardCost"]  = df["StandardCost"].fillna(0).round(2)
    df["Color"]         = df["Color"].fillna("N/A")
    df["ProductLine"]   = df["ProductLine"].fillna("N/A")
    log.info(f"Products: {len(df)} rows")
    return df


def load_stock_levels() -> pd.DataFrame:
    """On-hand quantity per product, one row for EVERY product.

    Inventory coverage has three distinct states, kept apart by the
    HasInventoryRecord flag so that "we have no data" can never be read as
    "we have none of it":

    1. Known inventory quantity — the product has ProductInventory rows, so
       TotalQuantity is a real figure and both flags are meaningful.
    2. Confirmed zero quantity — the product has inventory rows whose
       quantities sum to zero. TotalQuantity stays 0 and the product is
       legitimately flagged below safety stock.
    3. Missing inventory record — HasInventoryRecord is False,
       TotalQuantity stays MISSING (never 0) and both flags are False: an
       absent row says nothing about stock, so the product is neither
       healthy nor stocked out.

    The join is driven by Product rather than by the inventory aggregation,
    so products without inventory rows are reported instead of vanishing.
    On the canonical dataset all 504 products appear: 432 with a record (4 of
    them at zero) and 72 without. The flags only ever compare recorded
    quantities, so the extra rows do not change the alert counts — still 69
    below safety stock and 7 needing reorder.

    Count stock over the recorded subset (see
    recorded_stock_product_count, below_safety_product_count and
    needs_reorder_product_count), never over the whole frame as if every
    product had known stock.
    """
    log.info("Loading stock levels...")
    inventory = read_csv(MM_DIR / "ProductInventory.csv", "ProductInventory")
    products  = read_csv(MM_DIR / "Product.csv", "Product")[
        ["ProductID", "Name", "ProductNumber",
         "SafetyStockLevel", "ReorderPoint"]
    ]

    # Aggregate inventory quantity per product across all locations.
    # min_count keeps a group of empty quantities missing instead of
    # reporting it as zero on-hand, and the nullable integer dtype survives
    # the CSV round trip as "number" or "blank", never as 0.
    inv_agg = inventory.groupby("ProductID")["Quantity"].sum(min_count=1).reset_index()
    inv_agg.columns = ["ProductID", "TotalQuantity"]
    inv_agg["TotalQuantity"] = inv_agg["TotalQuantity"].convert_dtypes()

    df = products.merge(inv_agg, on="ProductID", how="left")

    # Membership of the inventory table, not a quantity test: a product with
    # rows that carry no quantity still HAS a record, its amount is just
    # unknown, and that must stay distinguishable from a confirmed zero.
    df["HasInventoryRecord"] = df["ProductID"].isin(set(inventory["ProductID"]))

    # Unknown quantity compares to nothing, so the flags stay False wherever
    # there is no recorded stock figure — no fabricated stock-out alerts.
    recorded = df["HasInventoryRecord"]
    df["BelowSafety"]  = (recorded & (df["TotalQuantity"] < df["SafetyStockLevel"])).fillna(False)
    df["NeedsReorder"] = (recorded & (df["TotalQuantity"] < df["ReorderPoint"])).fillna(False)

    df = clean_dates(df)
    log.info(f"Stock levels: {len(df)} rows "
             f"({int(recorded.sum())} with an inventory record, "
             f"{int((~recorded).sum())} without)")
    return df


def load_purchase_orders() -> pd.DataFrame:
    log.info("Loading purchase orders...")
    header  = read_csv(MM_DIR / "PurchaseOrderHeader.csv", "PurchaseOrderHeader")
    detail  = read_csv(MM_DIR / "PurchaseOrderDetail.csv", "PurchaseOrderDetail")
    product = read_csv(MM_DIR / "Product.csv", "Product")[
        ["ProductID", "Name", "ProductNumber"]
    ]

    detail = detail.merge(product, on="ProductID", how="left")
    df     = header.merge(
        detail[["PurchaseOrderID", "ProductID", "Name",
                "ProductNumber", "OrderQty", "UnitPrice", "LineTotal",
                "ReceivedQty", "RejectedQty", "StockedQty"]],
        on="PurchaseOrderID", how="left"
    )

    df = clean_dates(df)
    df = drop_unused(df)
    df["Status"] = df["Status"].map({
        1: "Pending", 2: "Approved",
        3: "Rejected", 4: "Complete"
    }).fillna("Unknown")
    df["TotalDue"] = df["TotalDue"].round(2)

    log.info(f"Purchase orders: {len(df)} rows")
    return df


def header_level_sum(df: pd.DataFrame, header_col: str, value_col: str) -> float:
    """Sum a HEADER-level column without repeating it for every detail line.

    load_sales_orders() and load_purchase_orders() left-join a header onto its
    detail lines, so a header-level amount (TotalDue, TaxAmt, Freight) is
    repeated once per line while staying constant inside the group.
    Deduplicating on the header key first restores the true header grain.
    """
    return float(df.drop_duplicates(subset=[header_col])[value_col].sum())


def header_level_count(df: pd.DataFrame, header_col: str, **filters) -> int:
    """Count distinct header records, optionally filtered on header-level columns.

    ``header_level_count(po, "PurchaseOrderID", Status="Pending")`` counts
    pending purchase ORDERS. Counting matching rows of the joined frame
    instead would count detail lines, which is a different question.
    """
    headers = df.drop_duplicates(subset=[header_col])
    for column, expected in filters.items():
        headers = headers[headers[column] == expected]
    return int(len(headers))


def order_level_revenue(sales: pd.DataFrame) -> float:
    """Revenue at ORDER grain — TotalDue counted once per SalesOrderID.

    load_sales_orders() left-joins the header onto the detail rows, so every
    order repeats its order-level TotalDue once per detail line while the
    value itself stays constant. Summing the joined column therefore
    double-counts revenue: on the full dataset the naive sum is
    USD 2,926,970,116.50 against a correct USD 123,216,779.55 — a factor of
    about 23.75x, not the 3.86x implied by 121,317 rows over 31,465 orders.
    The factor equals the value-WEIGHTED average number of detail lines per
    order, so it depends on how order values and line counts are distributed:
    high-value orders carry many more lines (up to 72) and are inflated
    disproportionately. Always use this helper — or LineTotal — never
    ``sales["TotalDue"].sum()``.
    """
    return header_level_sum(sales, "SalesOrderID", "TotalDue")


def line_level_revenue(sales: pd.DataFrame) -> float:
    """Revenue at LINE grain — sum of LineTotal across order-detail rows."""
    return float(sales["LineTotal"].sum())


def purchase_order_level_total(purchase_orders: pd.DataFrame) -> float:
    """Purchase spend at ORDER grain — TotalDue counted once per PurchaseOrderID.

    Same trap as sales: summing TotalDue over the 8,845 joined detail lines
    gives USD 180,741,103 while the 4,012 purchase orders total
    USD 70,479,332. The multiplier depends on order value against line count.
    """
    return header_level_sum(purchase_orders, "PurchaseOrderID", "TotalDue")


def purchase_line_total(purchase_orders: pd.DataFrame) -> float:
    """Purchase amounts at LINE grain — sum of LineTotal across PO detail rows."""
    return float(purchase_orders["LineTotal"].sum())


def pending_purchase_order_count(purchase_orders: pd.DataFrame) -> int:
    """Purchase ORDERS whose header Status is "Pending", counted at order grain.

    On the full dataset this is 225 orders; the joined frame contains 477
    detail lines belonging to those orders, which is not the same answer.
    """
    return header_level_count(purchase_orders, "PurchaseOrderID", Status="Pending")


def recorded_stock_product_count(stock: pd.DataFrame) -> int:
    """Products with an actual ProductInventory record.

    This is the honest denominator for any stock KPI: the frame also carries
    products whose quantity is simply unknown, and counting or averaging over
    them would present missing data as a measured stock level.
    """
    return int(stock["HasInventoryRecord"].sum())


def below_safety_product_count(stock: pd.DataFrame) -> int:
    """Products with recorded stock below their safety stock level.

    Products without an inventory record are excluded, so the count is not
    inflated by unknown quantities. On the full dataset it is 69 — the same
    figure as when unrecorded products were absent from the frame entirely.
    """
    recorded = stock[stock["HasInventoryRecord"]]
    return int(recorded["BelowSafety"].sum())


def needs_reorder_product_count(stock: pd.DataFrame) -> int:
    """Products with recorded stock below their reorder point (7 on the dataset)."""
    recorded = stock[stock["HasInventoryRecord"]]
    return int(recorded["NeedsReorder"].sum())


def load_and_save_all():
    log.info("Preflight validation...")
    validate_dataset()

    SD_DIR.mkdir(parents=True, exist_ok=True)
    MM_DIR.mkdir(parents=True, exist_ok=True)

    sales    = load_sales_orders()
    customers = load_customers()
    products  = load_products()
    stock     = load_stock_levels()
    po        = load_purchase_orders()

    sales.to_csv(SD_DIR     / "sales_orders_clean.csv",    index=False)
    customers.to_csv(SD_DIR / "customer_master_clean.csv", index=False)
    products.to_csv(MM_DIR  / "product_master_clean.csv",  index=False)
    stock.to_csv(MM_DIR     / "stock_levels_clean.csv",    index=False)
    po.to_csv(MM_DIR        / "purchase_orders_clean.csv", index=False)

    log.info("All clean files saved.")
    log.info(f"  Sales orders:    {len(sales)}")
    log.info(f"  Customers:       {len(customers)}")
    log.info(f"  Products:        {len(products)}")
    log.info(f"  Stock levels:    {len(stock)}")
    log.info(f"  Purchase orders: {len(po)}")
    log.info(f"  Below safety stock: {below_safety_product_count(stock)} "
             f"(of {recorded_stock_product_count(stock)} products with recorded stock)")


if __name__ == "__main__":
    load_and_save_all()