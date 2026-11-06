"""Inventory coverage: every product is reported, unknown is never zero.

`load_stock_levels()` is driven by the product master, so the 72 AdventureWorks
products that have no `ProductInventory` row at all stay visible with
`HasInventoryRecord = False` instead of disappearing from the cleaned table.
The three inventory states must remain distinguishable:

* recorded quantity   — real on-hand figure, flags meaningful;
* recorded zero      — genuinely stocked out, legitimately below safety stock;
* no inventory record — quantity unknown, must never be read as 0 or alerted.

Fixture tests below pin the semantics; the final tests re-check the figures the
README promises on the real dataset and are skipped when it has not been loaded.
"""
import pandas as pd
import pytest

from src.chunking.chunker import MISSING_PLACEHOLDER, chunk_table
from src.data import loader
from src.data.loader import (
    below_safety_product_count,
    needs_reorder_product_count,
    recorded_stock_product_count,
)
from tests.conftest import sample_row, write_table

# ProductID, SafetyStockLevel, ReorderPoint, per-location quantities
# (None means the product has no ProductInventory row at all).
PRODUCTS = [
    (1, 10, 20, [3, 2]),     # recorded, below safety stock and reorder point
    (2, 10, 20, [30]),       # recorded, comfortably stocked
    (3, 4,  3,  [0, 0]),     # recorded, confirmed zero on hand
    (4, 10, 20, None),       # no inventory record
    (5, 10, 20, None),       # no inventory record
]

TOTAL_PRODUCTS         = len(PRODUCTS)
RECORDED_PRODUCTS      = 3
UNRECORDED_PRODUCTS    = 2
BELOW_SAFETY_PRODUCTS  = 2   # products 1 and 3 — NOT 4 (products 4 and 5 are unknown)
NEEDS_REORDER_PRODUCTS = 2
ZERO_QUANTITY_PRODUCTS = 1   # product 3


@pytest.fixture
def coverage_dataset(aw_dataset):
    """The default mini dataset, rewritten with mixed inventory coverage."""
    write_table(aw_dataset / "mm" / "Product.csv", "Product", [
        sample_row("Product", ProductID=product_id, Name=f"Product {product_id}",
                   SafetyStockLevel=safety, ReorderPoint=reorder)
        for product_id, safety, reorder, _ in PRODUCTS
    ])
    write_table(aw_dataset / "mm" / "ProductInventory.csv", "ProductInventory", [
        sample_row("ProductInventory", ProductID=product_id, Quantity=quantity,
                   LocationID=location)
        for product_id, _, _, quantities in PRODUCTS
        for location, quantity in enumerate(quantities or [], start=1)
    ])
    return aw_dataset


def legacy_stock_frame() -> pd.DataFrame:
    """The pre-coverage calculation: inventory aggregation drives the join.

    Used as the oracle that recorded products must keep exactly the quantities
    and flags it produced before products without a record were added.
    """
    inventory = loader.read_csv(loader.MM_DIR / "ProductInventory.csv",
                                "ProductInventory")
    products  = loader.read_csv(loader.MM_DIR / "Product.csv", "Product")[
        ["ProductID", "SafetyStockLevel", "ReorderPoint"]
    ]
    aggregated = inventory.groupby("ProductID")["Quantity"].sum().reset_index()
    aggregated.columns = ["ProductID", "TotalQuantity"]

    frame = aggregated.merge(products, on="ProductID", how="left")
    frame["BelowSafety"]  = frame["TotalQuantity"] < frame["SafetyStockLevel"]
    frame["NeedsReorder"] = frame["TotalQuantity"] < frame["ReorderPoint"]
    return frame.sort_values("ProductID").reset_index(drop=True)


def as_ordered_records(frame: pd.DataFrame, columns: list) -> list:
    """Row-order-independent [{column: value}] view for comparing frames."""
    return frame.sort_values("ProductID")[columns].to_dict(orient="records")


def row_for(stock: pd.DataFrame, product_id: int) -> pd.Series:
    matches = stock[stock["ProductID"] == product_id]
    assert len(matches) == 1, f"product {product_id} appears {len(matches)} times"
    return matches.iloc[0]


# ── coverage: all products represented ───────────────────────────────────
def test_every_product_appears_exactly_once(coverage_dataset):
    stock = loader.load_stock_levels()
    assert len(stock) == TOTAL_PRODUCTS
    assert set(stock["ProductID"]) == {p[0] for p in PRODUCTS}
    assert not stock["ProductID"].duplicated().any()


def test_products_are_split_into_recorded_and_unrecorded(coverage_dataset):
    stock = loader.load_stock_levels()
    assert int(stock["HasInventoryRecord"].sum()) == RECORDED_PRODUCTS
    assert int((~stock["HasInventoryRecord"]).sum()) == UNRECORDED_PRODUCTS
    assert recorded_stock_product_count(stock) == RECORDED_PRODUCTS


@pytest.mark.parametrize("product_id, recorded", [
    (1, True), (2, True), (3, True), (4, False), (5, False),
])
def test_inventory_record_flag_reflects_presence_of_inventory_rows(
        coverage_dataset, product_id, recorded):
    assert bool(row_for(loader.load_stock_levels(), product_id)["HasInventoryRecord"]) is recorded


def test_unrecorded_products_keep_a_missing_quantity(coverage_dataset):
    """Unknown must stay unknown: the cell may not be filled with zero."""
    stock = loader.load_stock_levels()
    for product_id in (4, 5):
        row = row_for(stock, product_id)
        assert pd.isna(row["TotalQuantity"])
        assert bool(row["HasInventoryRecord"]) is False

    # Comparing an unknown quantity against 0 yields no answer at all, so the
    # count below is the direct proof that nothing was silently turned into a
    # stock-out: only the genuinely-zero product is reported as zero.
    assert int((stock["TotalQuantity"] == 0).sum()) == ZERO_QUANTITY_PRODUCTS


def test_unrecorded_products_raise_no_alerts(coverage_dataset):
    stock = loader.load_stock_levels()
    unrecorded = stock[~stock["HasInventoryRecord"]]
    assert not unrecorded.empty
    assert int(unrecorded["BelowSafety"].sum()) == 0
    assert int(unrecorded["NeedsReorder"].sum()) == 0


def test_confirmed_zero_stock_stays_zero_and_still_alerts(coverage_dataset):
    """A recorded quantity of 0 is data, not a gap: it must survive as 0."""
    row = row_for(loader.load_stock_levels(), 3)
    assert bool(row["HasInventoryRecord"]) is True
    assert float(row["TotalQuantity"]) == 0.0
    assert pd.notna(row["TotalQuantity"])
    assert bool(row["BelowSafety"]) is True
    assert bool(row["NeedsReorder"]) is True


# ── alert counts must not be inflated by unknown quantities ──────────────
def test_alert_counts_cover_recorded_stock_only(coverage_dataset):
    stock = loader.load_stock_levels()
    assert below_safety_product_count(stock)  == BELOW_SAFETY_PRODUCTS
    assert needs_reorder_product_count(stock) == NEEDS_REORDER_PRODUCTS


def test_treating_missing_records_as_zero_would_fabricate_alerts(coverage_dataset):
    """Pins the failure mode the flag prevents: 0 < SafetyStockLevel for 4 and 5."""
    stock = loader.load_stock_levels()
    naive_zero_stock = stock["TotalQuantity"].fillna(0)

    invented = int((naive_zero_stock < stock["SafetyStockLevel"]).sum())
    assert invented == BELOW_SAFETY_PRODUCTS + UNRECORDED_PRODUCTS
    assert invented > below_safety_product_count(stock)


def test_helpers_match_an_explicit_recorded_filter(coverage_dataset):
    stock = loader.load_stock_levels()
    recorded = stock[stock["HasInventoryRecord"]]
    assert below_safety_product_count(stock)  == int(recorded["BelowSafety"].sum())
    assert needs_reorder_product_count(stock) == int(recorded["NeedsReorder"].sum())
    assert recorded_stock_product_count(stock) == len(recorded)


# ── existing calculations unchanged ──────────────────────────────────────
def test_recorded_rows_are_identical_to_the_previous_calculation(coverage_dataset):
    """Products that were already reported keep their quantity and flags."""
    stock   = loader.load_stock_levels()
    columns = ["ProductID", "TotalQuantity", "SafetyStockLevel", "ReorderPoint",
               "BelowSafety", "NeedsReorder"]

    expected = as_ordered_records(legacy_stock_frame(), columns)
    actual   = as_ordered_records(stock[stock["HasInventoryRecord"]], columns)

    for before, after in zip(expected, actual):
        assert before["ProductID"] == after["ProductID"]
        assert float(before["TotalQuantity"]) == float(after["TotalQuantity"])
        assert bool(before["BelowSafety"]) is bool(after["BelowSafety"])
        assert bool(before["NeedsReorder"]) is bool(after["NeedsReorder"])
    assert len(actual) == RECORDED_PRODUCTS


def test_recorded_alerts_still_use_the_real_thresholds(coverage_dataset):
    stock = loader.load_stock_levels()
    product_1 = row_for(stock, 1)
    assert float(product_1["TotalQuantity"]) == 5          # 3 + 2 across locations
    assert bool(product_1["BelowSafety"]) is True          # 5 < 10
    assert bool(product_1["NeedsReorder"]) is True         # 5 < 20
    product_2 = row_for(stock, 2)
    assert bool(product_2["BelowSafety"]) is False         # 30 >= 10
    assert bool(product_2["NeedsReorder"]) is False        # 30 >= 20


# ── persisted artefact keeps the distinction ────────────────────────────
@pytest.fixture
def persisted_stock(coverage_dataset):
    loader.load_and_save_all()
    path = loader.MM_DIR / "stock_levels_clean.csv"
    return pd.read_csv(path), path.read_text(encoding="utf-8").splitlines()


def test_clean_csv_round_trip_keeps_unknown_quantity_missing(persisted_stock):
    stock, _ = persisted_stock
    assert len(stock) == TOTAL_PRODUCTS
    assert "HasInventoryRecord" in stock.columns
    assert below_safety_product_count(stock)  == BELOW_SAFETY_PRODUCTS
    assert needs_reorder_product_count(stock) == NEEDS_REORDER_PRODUCTS

    unrecorded = stock[~stock["HasInventoryRecord"]]
    assert len(unrecorded) == UNRECORDED_PRODUCTS
    assert unrecorded["TotalQuantity"].isna().all()
    assert not unrecorded["BelowSafety"].any()
    assert not unrecorded["NeedsReorder"].any()


def test_clean_csv_writes_an_empty_cell_not_a_zero_for_unknown_stock(persisted_stock):
    _, lines = persisted_stock
    header = lines[0].split(",")
    quantity_column = header.index("TotalQuantity")
    id_column = header.index("ProductID")

    unknown_rows = [line.split(",") for line in lines[1:]
                    if line.split(",")[id_column] in {"4", "5"}]
    assert len(unknown_rows) == UNRECORDED_PRODUCTS
    assert all(fields[quantity_column] == "" for fields in unknown_rows)


def test_zero_quantity_survives_the_csv_round_trip(persisted_stock):
    stock, _ = persisted_stock
    zero_rows = stock[stock["TotalQuantity"] == 0]
    assert len(zero_rows) == ZERO_QUANTITY_PRODUCTS
    assert zero_rows["HasInventoryRecord"].tolist() == [True]
    assert zero_rows["BelowSafety"].tolist() == [True]


# ── chunk text (item 3) ──────────────────────────────────────────────────
def field_pairs(text: str) -> dict:
    body = text.split("] ", 1)[1]
    return {field.split(": ", 1)[0]: field.split(": ", 1)[1]
            for field in body.split(" | ")}


@pytest.fixture
def stock_chunks(coverage_dataset):
    return chunk_table(loader.load_stock_levels(), "stock_levels", "ProductID")


def test_missing_stock_quantity_renders_as_unknown_in_chunk_text(stock_chunks):
    by_id = {c["metadata"]["record_id"]: c["text"] for c in stock_chunks}

    for product_id in ("4", "5"):
        pairs = field_pairs(by_id[product_id])
        assert pairs["TotalQuantity"] == MISSING_PLACEHOLDER
        assert pairs["HasInventoryRecord"] == "False"
        assert pairs["BelowSafety"] == "False"
        assert pairs["NeedsReorder"] == "False"


def test_recorded_and_zero_quantities_render_as_numbers(stock_chunks):
    by_id = {c["metadata"]["record_id"]: c["text"] for c in stock_chunks}

    assert float(field_pairs(by_id["1"])["TotalQuantity"]) == 5.0
    assert field_pairs(by_id["4"])["TotalQuantity"] == MISSING_PLACEHOLDER
    zero = field_pairs(by_id["3"])["TotalQuantity"]
    assert zero != MISSING_PLACEHOLDER
    assert float(zero) == 0.0


def test_no_stock_chunk_leaks_a_missing_value_token(stock_chunks):
    leaked = [c["text"] for c in stock_chunks
              if any(value in {"nan", "NaN", "None", "NaT", "<NA>", ""}
                     for value in field_pairs(c["text"]).values())]
    assert not leaked, f"{len(leaked)} stock chunks leak a missing-value token"


# ── real dataset figures promised by the README ──────────────────────────
EXPECTED_REAL = {
    "products":           504,
    "recorded":           432,
    "unrecorded":          72,
    "below_safety":        69,
    "needs_reorder":        7,
    "zero_quantity":        4,
}


@pytest.fixture
def real_stock():
    path = loader.MM_DIR / "stock_levels_clean.csv"
    if not path.exists():
        pytest.skip("stock_levels_clean.csv not generated yet")
    frame = pd.read_csv(path)
    if "HasInventoryRecord" not in frame.columns:
        pytest.skip("clean output predates the coverage flag; re-run the loader")
    return frame


def test_real_dataset_coverage_matches_the_documented_counts(real_stock):
    stock = real_stock
    assert len(stock) == EXPECTED_REAL["products"]
    assert recorded_stock_product_count(stock)   == EXPECTED_REAL["recorded"]
    assert int((~stock["HasInventoryRecord"]).sum()) == EXPECTED_REAL["unrecorded"]
    assert below_safety_product_count(stock)     == EXPECTED_REAL["below_safety"]
    assert needs_reorder_product_count(stock)    == EXPECTED_REAL["needs_reorder"]
    assert int((stock["TotalQuantity"] == 0).sum()) == EXPECTED_REAL["zero_quantity"]


def test_real_dataset_unrecorded_products_are_neither_zero_nor_alerted(real_stock):
    unrecorded = real_stock[~real_stock["HasInventoryRecord"]]
    assert len(unrecorded) == EXPECTED_REAL["unrecorded"]
    assert unrecorded["TotalQuantity"].isna().all()
    assert not unrecorded["BelowSafety"].any()
    assert not unrecorded["NeedsReorder"].any()


def test_real_recorded_stock_calculations_are_unchanged(real_stock):
    """The 432 products reported before the change produce identical values."""
    expected = legacy_stock_frame()
    columns  = ["ProductID", "TotalQuantity", "SafetyStockLevel", "ReorderPoint",
                "BelowSafety", "NeedsReorder"]

    assert len(expected) == EXPECTED_REAL["recorded"]
    assert as_ordered_records(expected, columns) == as_ordered_records(
        real_stock[real_stock["HasInventoryRecord"]], columns)


def test_real_alert_counts_would_be_inflated_by_zero_fill(real_stock):
    stock = real_stock
    naive = int((stock["TotalQuantity"].fillna(0) < stock["SafetyStockLevel"]).sum())
    assert naive == EXPECTED_REAL["below_safety"] + EXPECTED_REAL["unrecorded"]
    assert naive > below_safety_product_count(stock)
