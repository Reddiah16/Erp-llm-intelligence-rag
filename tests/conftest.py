"""Shared fixtures: tiny AdventureWorks-shaped TSV files.

Every test uses these temporary fixtures instead of the real dataset, so the
suite never needs a download. The shape mirrors the canonical
oltp-install-script files: tab-delimited, no header row, and exactly the
number of columns declared in loader.HEADERS.
"""
from pathlib import Path

import pytest

from src.data import loader
from src.chunking import chunker

DATE_COLUMNS = {c for cols in loader.HEADERS.values() for c in cols if "Date" in c}

# Minimal but internally consistent dataset: one product, one customer,
# one sales order, one purchase order. SafetyStockLevel is deliberately above
# total on-hand so BelowSafety / NeedsReorder evaluate True.
STOCK_LEVEL     = 10
REORDER_POINT   = 20
ON_HAND_QUANTITY = 1


def sample_row(name: str, **overrides) -> list:
    """Build one row for table `name` in HEADERS order."""
    values = []
    for col in loader.HEADERS[name]:
        if col in overrides:
            values.append(str(overrides[col]))
        elif col in DATE_COLUMNS:
            values.append("2024-01-05 00:00:00.000")
        else:
            values.append("1")
    return values


def write_table(path: Path, name: str, rows: list) -> Path:
    """Write rows as tab-delimited, headerless lines."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        for row in rows:
            fh.write("\t".join(row) + "\n")
    return path


def default_rows(name: str) -> list:
    """One sensible default row per raw table."""
    if name == "Product":
        return [sample_row(name, SafetyStockLevel=STOCK_LEVEL,
                           ReorderPoint=REORDER_POINT)]
    if name == "ProductInventory":
        return [sample_row(name, Quantity=ON_HAND_QUANTITY)]
    return [sample_row(name)]


def build_raw_dataset(data_dir: Path) -> Path:
    """Write all seven required files into data_dir/sd and data_dir/mm."""
    for name, path in _raw_paths(data_dir).items():
        write_table(path, name, default_rows(name))
    return data_dir


def _raw_paths(data_dir: Path) -> dict:
    return {
        "SalesOrderHeader":    data_dir / "sd" / "SalesOrderHeader.csv",
        "SalesOrderDetail":    data_dir / "sd" / "SalesOrderDetail.csv",
        "Customer":            data_dir / "sd" / "Customer.csv",
        "Product":             data_dir / "mm" / "Product.csv",
        "ProductInventory":    data_dir / "mm" / "ProductInventory.csv",
        "PurchaseOrderHeader": data_dir / "mm" / "PurchaseOrderHeader.csv",
        "PurchaseOrderDetail": data_dir / "mm" / "PurchaseOrderDetail.csv",
    }


def point_modules_at(data_dir: Path, monkeypatch) -> None:
    """Make loader and chunker read/write inside `data_dir`."""
    sd, mm = data_dir / "sd", data_dir / "mm"
    monkeypatch.setattr(loader, "SD_DIR", sd)
    monkeypatch.setattr(loader, "MM_DIR", mm)
    monkeypatch.setattr(chunker, "SD_DIR", sd)
    monkeypatch.setattr(chunker, "MM_DIR", mm)


@pytest.fixture
def aw_dataset(tmp_path, monkeypatch) -> Path:
    """A complete, valid miniature dataset wired into loader + chunker."""
    data_dir = tmp_path / "data"
    build_raw_dataset(data_dir)
    point_modules_at(data_dir, monkeypatch)
    return data_dir


@pytest.fixture
def empty_data_dir(tmp_path, monkeypatch) -> Path:
    """Empty data directories: every required file is missing."""
    data_dir = tmp_path / "data"
    (data_dir / "sd").mkdir(parents=True)
    (data_dir / "mm").mkdir(parents=True)
    point_modules_at(data_dir, monkeypatch)
    return data_dir
