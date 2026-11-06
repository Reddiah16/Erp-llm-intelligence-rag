"""Regression guard for order-level vs line-level grain.

load_sales_orders() and load_purchase_orders() both left-join a header onto its
detail lines, so a header value such as TotalDue or Status is repeated on every
detail row. Summing or counting the joined column inflates the answer. These
tests pin the two grains apart for sales and purchase orders and fail if a
naive aggregation is reintroduced into the pipeline.
"""
import pytest

from src.data import loader
from src.data.loader import (
    header_level_count,
    header_level_sum,
    line_level_revenue,
    order_level_revenue,
    pending_purchase_order_count,
    purchase_line_total,
    purchase_order_level_total,
)
from tests.conftest import sample_row, write_table

# Order 1: two detail lines (100 + 50), order-level TotalDue 200.
# Order 2: one detail line (150), order-level TotalDue 150.
ORDER_LEVEL_TOTAL = 350.0     # 200 + 150, each counted once
LINE_LEVEL_TOTAL  = 300.0     # 100 + 50 + 150
NAIVE_JOINED_TOTAL = 550.0    # 200 + 200 + 150 — the double-counted figure


@pytest.fixture
def grain_data(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"

    write_table(data_dir / "sd" / "SalesOrderHeader.csv", "SalesOrderHeader", [
        sample_row("SalesOrderHeader", SalesOrderID=1, TotalDue=200),
        sample_row("SalesOrderHeader", SalesOrderID=2, TotalDue=150),
    ])
    write_table(data_dir / "sd" / "SalesOrderDetail.csv", "SalesOrderDetail", [
        sample_row("SalesOrderDetail", SalesOrderID=1, SalesOrderDetailID=1, LineTotal=100),
        sample_row("SalesOrderDetail", SalesOrderID=1, SalesOrderDetailID=2, LineTotal=50),
        sample_row("SalesOrderDetail", SalesOrderID=2, SalesOrderDetailID=3, LineTotal=150),
    ])
    write_table(data_dir / "mm" / "Product.csv", "Product", [sample_row("Product")])

    monkeypatch.setattr(loader, "SD_DIR", data_dir / "sd")
    monkeypatch.setattr(loader, "MM_DIR", data_dir / "mm")
    return data_dir


def test_joined_frame_keeps_detail_grain(grain_data):
    sales = loader.load_sales_orders()
    assert len(sales) == 3                      # one row per detail line
    assert sales["SalesOrderID"].nunique() == 2  # but only two orders exist


def test_order_level_revenue_counts_totaldue_once_per_order(grain_data):
    sales = loader.load_sales_orders()
    assert order_level_revenue(sales) == pytest.approx(ORDER_LEVEL_TOTAL)


def test_line_level_revenue_sums_detail_lines(grain_data):
    sales = loader.load_sales_orders()
    assert line_level_revenue(sales) == pytest.approx(LINE_LEVEL_TOTAL)


def test_the_two_grains_are_not_interchangeable(grain_data):
    sales = loader.load_sales_orders()
    assert order_level_revenue(sales) != line_level_revenue(sales)


def test_naive_totaldue_sum_double_counts_revenue(grain_data):
    """Pins the exact failure mode: summing TotalDue over joined rows."""
    sales = loader.load_sales_orders()
    naive_sum = float(sales["TotalDue"].sum())

    assert naive_sum == pytest.approx(NAIVE_JOINED_TOTAL)
    assert naive_sum > order_level_revenue(sales)


def test_order_level_revenue_matches_deduped_sum(grain_data):
    sales = loader.load_sales_orders()
    expected = float(sales.drop_duplicates(subset=["SalesOrderID"])["TotalDue"].sum())
    assert order_level_revenue(sales) == pytest.approx(expected)


# ── Purchase orders ──────────────────────────────────────────────────────
# PO 1 is Pending with two detail lines; PO 2 is Complete with one.
# Order grain: 350 total, 1 pending order. Line grain: 300. Naive joined
# TotalDue sum: 550. Naive pending-row count: 2, which is the bug this guards.
PO_ORDER_LEVEL_TOTAL  = 350.0
PO_LINE_LEVEL_TOTAL   = 300.0
PO_NAIVE_JOINED_TOTAL = 550.0
PO_PENDING_ORDERS     = 1
PO_PENDING_LINES      = 2


@pytest.fixture
def po_grain_data(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"

    write_table(data_dir / "mm" / "PurchaseOrderHeader.csv", "PurchaseOrderHeader", [
        sample_row("PurchaseOrderHeader", PurchaseOrderID=1, Status=1, TotalDue=200),
        sample_row("PurchaseOrderHeader", PurchaseOrderID=2, Status=4, TotalDue=150),
    ])
    write_table(data_dir / "mm" / "PurchaseOrderDetail.csv", "PurchaseOrderDetail", [
        sample_row("PurchaseOrderDetail", PurchaseOrderID=1,
                   PurchaseOrderDetailID=1, LineTotal=100),
        sample_row("PurchaseOrderDetail", PurchaseOrderID=1,
                   PurchaseOrderDetailID=2, LineTotal=50),
        sample_row("PurchaseOrderDetail", PurchaseOrderID=2,
                   PurchaseOrderDetailID=3, LineTotal=150),
    ])
    write_table(data_dir / "mm" / "Product.csv", "Product", [sample_row("Product")])

    monkeypatch.setattr(loader, "SD_DIR", data_dir / "sd")
    monkeypatch.setattr(loader, "MM_DIR", data_dir / "mm")
    return data_dir


def test_purchase_orders_frame_keeps_detail_grain(po_grain_data):
    po = loader.load_purchase_orders()
    assert len(po) == 3                        # one row per detail line
    assert po["PurchaseOrderID"].nunique() == 2  # but only two orders exist


def test_purchase_order_level_total_counts_totaldue_once(po_grain_data):
    po = loader.load_purchase_orders()
    assert purchase_order_level_total(po) == pytest.approx(PO_ORDER_LEVEL_TOTAL)
    naive = float(po["TotalDue"].sum())
    assert naive == pytest.approx(PO_NAIVE_JOINED_TOTAL)
    assert naive > purchase_order_level_total(po)


def test_purchase_line_total_is_a_different_grain(po_grain_data):
    po = loader.load_purchase_orders()
    assert purchase_line_total(po) == pytest.approx(PO_LINE_LEVEL_TOTAL)
    assert purchase_line_total(po) != purchase_order_level_total(po)


def test_pending_purchase_order_count_counts_orders_not_lines(po_grain_data):
    """The dashboard KPI bug: 2 pending detail lines belong to 1 pending order."""
    po = loader.load_purchase_orders()
    assert pending_purchase_order_count(po) == PO_PENDING_ORDERS
    assert int((po["Status"] == "Pending").sum()) == PO_PENDING_LINES


def test_header_level_helpers_reproduce_the_named_helpers(po_grain_data, grain_data):
    """The generic helper is the single implementation behind both grains."""
    po    = loader.load_purchase_orders()
    sales = loader.load_sales_orders()
    assert header_level_sum(po, "PurchaseOrderID", "TotalDue") == \
        pytest.approx(purchase_order_level_total(po))
    assert header_level_sum(sales, "SalesOrderID", "TotalDue") == \
        pytest.approx(order_level_revenue(sales))
    assert header_level_count(po, "PurchaseOrderID", Status="Pending") == \
        pending_purchase_order_count(po)
