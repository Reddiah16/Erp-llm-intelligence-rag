import pandas as pd
import pytest
from pathlib import Path

DATA_AVAILABLE = Path("data/sd/sales_orders_clean.csv").exists()


def test_loader_imports():
    from src.data.loader import (
        load_sales_orders, load_customers,
        load_products, load_stock_levels,
        load_purchase_orders
    )
    assert callable(load_sales_orders)
    assert callable(load_customers)
    assert callable(load_products)
    assert callable(load_stock_levels)
    assert callable(load_purchase_orders)


@pytest.mark.skipif(not DATA_AVAILABLE, reason="AdventureWorks data not available in CI")
def test_clean_files_exist():
    assert Path("data/sd/sales_orders_clean.csv").exists()
    assert Path("data/sd/customer_master_clean.csv").exists()
    assert Path("data/mm/product_master_clean.csv").exists()
    assert Path("data/mm/stock_levels_clean.csv").exists()
    assert Path("data/mm/purchase_orders_clean.csv").exists()


@pytest.mark.skipif(not DATA_AVAILABLE, reason="AdventureWorks data not available in CI")
def test_sales_orders_columns():
    df = pd.read_csv("data/sd/sales_orders_clean.csv")
    assert "SalesOrderID" in df.columns
    assert "TotalDue"     in df.columns
    assert "Status"       in df.columns
    assert "CustomerID"   in df.columns


@pytest.mark.skipif(not DATA_AVAILABLE, reason="AdventureWorks data not available in CI")
def test_stock_levels_has_safety_flag():
    df = pd.read_csv("data/mm/stock_levels_clean.csv")
    assert "BelowSafety"  in df.columns
    assert "NeedsReorder" in df.columns
    assert df["BelowSafety"].sum() > 0


@pytest.mark.skipif(not DATA_AVAILABLE, reason="AdventureWorks data not available in CI")
def test_products_have_names():
    df = pd.read_csv("data/mm/product_master_clean.csv")
    assert "Name"          in df.columns
    assert "ProductNumber" in df.columns
    assert df["Name"].notna().all()


@pytest.mark.skipif(not DATA_AVAILABLE, reason="AdventureWorks data not available in CI")
def test_purchase_orders_status_readable():
    df = pd.read_csv("data/mm/purchase_orders_clean.csv")
    assert "Status" in df.columns
    valid = {"Pending", "Approved", "Rejected", "Complete", "Unknown"}
    assert set(df["Status"].unique()).issubset(valid)