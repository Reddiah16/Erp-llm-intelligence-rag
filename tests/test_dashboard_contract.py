"""Data-contract tests: the dashboard and API must speak the loader's schema.

These guard the Phase 1 decision that the AdventureWorks loader output is the
canonical contract, and that synthetic generator artefacts (regions, material
IDs, EUR amounts, "Open" purchase orders) cannot leak back into the app.
"""
import ast
from pathlib import Path

import pandas as pd
import pytest

from src.chunking import chunker
from src.data import loader
from src.data.loader import (
    below_safety_product_count,
    needs_reorder_product_count,
    pending_purchase_order_count,
    recorded_stock_product_count,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DASHBOARD = REPO_ROOT / "dashboard" / "app.py"
API_MAIN  = REPO_ROOT / "api" / "main.py"
PROMPTS   = REPO_ROOT / "src" / "llm" / "prompts.py"

CANONICAL_TABLES = {"sales_orders", "customer_master", "product_master",
                    "stock_levels", "purchase_orders"}

# Every column the dashboard reads, per cleaned file.
DASHBOARD_CONTRACT = {
    "sales_orders_clean.csv":    ["SalesOrderID", "TotalDue", "LineTotal", "Name"],
    "stock_levels_clean.csv":    ["Name", "TotalQuantity", "SafetyStockLevel",
                                  "ReorderPoint", "BelowSafety",
                                  "HasInventoryRecord"],
    "purchase_orders_clean.csv": ["PurchaseOrderID", "Status", "TotalDue"],
}

# Identifiers that only exist in the synthetic generator schema.
FORBIDDEN_DASHBOARD_TOKENS = [
    "total_value", "below_safety", "current_stock", "safety_stock",
    "material_id", "material_master", "region", "EUR",
    '== "Open"', "generator",
]


def _assigned_dict(path: Path, var_name: str) -> dict:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == var_name:
                    return ast.literal_eval(node.value)
    raise AssertionError(f"{var_name} not found in {path}")


@pytest.fixture
def clean_outputs(aw_dataset, monkeypatch, tmp_path):
    """Build the cleaned CSVs, from an unrelated CWD, and return their dir."""
    monkeypatch.chdir(tmp_path)
    loader.load_and_save_all()
    return aw_dataset


def test_dashboard_reads_the_cleaned_loader_files():
    source = DASHBOARD.read_text(encoding="utf-8")
    for filename in DASHBOARD_CONTRACT:
        assert filename in source, f"dashboard does not reference {filename}"


def test_clean_outputs_provide_every_column_the_dashboard_needs(clean_outputs):
    for filename, columns in DASHBOARD_CONTRACT.items():
        subdir = "sd" if filename.startswith(("sales", "customer")) else "mm"
        df = pd.read_csv(clean_outputs / subdir / filename)
        missing = set(columns) - set(df.columns)
        assert not missing, f"{filename} is missing {missing}"


def test_dashboard_status_values_match_loader_labels(clean_outputs):
    sales = pd.read_csv(clean_outputs / "sd" / "sales_orders_clean.csv")
    po    = pd.read_csv(clean_outputs / "mm" / "purchase_orders_clean.csv")

    # Status codes are mapped to labels by the loader, never raw integers.
    assert set(sales["Status"].unique()) <= {
        "In Process", "Approved", "Backordered", "Rejected", "Shipped", "Cancelled", "Unknown"}
    assert set(po["Status"].unique()) <= {
        "Pending", "Approved", "Rejected", "Complete", "Unknown"}
    assert "Open" not in set(po["Status"].unique())


def test_stock_alerts_are_computed_on_clean_data(clean_outputs):
    stock = pd.read_csv(clean_outputs / "mm" / "stock_levels_clean.csv")
    assert stock["HasInventoryRecord"].all()
    assert stock["BelowSafety"].sum() >= 1


def test_dashboard_computes_stock_kpis_over_recorded_inventory_only():
    """Guards the coverage bug: products without an inventory record have
    unknown stock, so they may not enter the stock KPIs as if they were zero."""
    source = DASHBOARD.read_text(encoding="utf-8")
    assert 'stock[stock["HasInventoryRecord"]]' in source
    assert 'stock[~stock["HasInventoryRecord"]]' in source
    assert "int(stock['BelowSafety'].sum())" not in source
    # The distinction has to be visible, not merely computed correctly.
    assert "ProductInventory" in source
    assert "unknown" in source.lower()


def test_stock_helpers_match_the_dashboard_expressions(clean_outputs):
    stock = pd.read_csv(clean_outputs / "mm" / "stock_levels_clean.csv")
    recorded   = stock[stock["HasInventoryRecord"]]
    unrecorded = stock[~stock["HasInventoryRecord"]]

    assert recorded_stock_product_count(stock) == len(recorded)
    assert below_safety_product_count(stock) == int(recorded["BelowSafety"].sum())
    assert needs_reorder_product_count(stock) == int(recorded["NeedsReorder"].sum())
    # Unknown stock never surfaces as an alert.
    assert int(unrecorded["BelowSafety"].sum()) == 0
    assert int(unrecorded["NeedsReorder"].sum()) == 0


def test_dashboard_has_no_synthetic_schema_dependencies():
    source = DASHBOARD.read_text(encoding="utf-8")
    for token in FORBIDDEN_DASHBOARD_TOKENS:
        assert token not in source, f"dashboard still references {token!r}"


def test_dashboard_does_not_sum_order_level_totaldue_over_detail_rows():
    source = DASHBOARD.read_text(encoding="utf-8")
    assert "drop_duplicates" in source
    assert "sales['TotalDue'].sum()" not in source


def test_dashboard_counts_pending_purchase_orders_at_order_grain():
    """Guards the KPI bug: pending DETAIL lines are not pending ORDERS."""
    source = DASHBOARD.read_text(encoding="utf-8")
    assert 'drop_duplicates(subset=["PurchaseOrderID"])' in source
    assert "(po['Status'] == 'Pending').sum()" not in source


def test_pending_po_helper_matches_the_dashboard_expression(clean_outputs):
    po = pd.read_csv(clean_outputs / "mm" / "purchase_orders_clean.csv")
    dashboard_expression = int((
        po.drop_duplicates(subset=["PurchaseOrderID"])["Status"] == "Pending"
    ).sum())
    assert pending_purchase_order_count(po) == dashboard_expression


def test_api_table_keywords_only_point_at_canonical_tables():
    mapping = _assigned_dict(API_MAIN, "TABLE_KEYWORDS")
    assert set(mapping.values()) <= CANONICAL_TABLES
    assert "material_master" not in mapping.values()


def test_retrieved_tables_are_scoreable_by_the_api(clean_outputs):
    """Chunker table names must cover what infer_relevant_table can return."""
    mapping = _assigned_dict(API_MAIN, "TABLE_KEYWORDS")
    chunks = chunker.load_and_chunk_all()
    produced = {c["metadata"]["table"] for c in chunks}

    assert produced == CANONICAL_TABLES
    assert set(mapping.values()) <= produced


def test_rag_prompt_carries_dataset_currency_not_a_hardcoded_one():
    source = PROMPTS.read_text(encoding="utf-8")
    assert "EUR" not in source
    assert "USD" in source
    assert "Never convert" in source


def test_generator_is_not_part_of_the_application_workflow():
    runtime_files = [
        *list((REPO_ROOT / "api").rglob("*.py")),
        *list((REPO_ROOT / "dashboard").rglob("*.py")),
        *[p for p in (REPO_ROOT / "src").rglob("*.py")
          if p.name != "generator.py"],
    ]
    for path in runtime_files:
        assert "data.generator" not in path.read_text(encoding="utf-8"), \
            f"{path.name} still imports the synthetic generator"


def test_generator_module_is_preserved():
    assert (REPO_ROOT / "src" / "data" / "generator.py").exists()
