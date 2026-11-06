"""Path anchoring: the pipeline must not depend on the terminal's CWD."""
from pathlib import Path

import pytest

from src.chunking import chunker
from src.data import loader

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_loader_paths_are_anchored_to_repo_root():
    assert loader.ROOT_DIR == REPO_ROOT
    assert loader.SD_DIR == REPO_ROOT / "data" / "sd"
    assert loader.MM_DIR == REPO_ROOT / "data" / "mm"


def test_chunker_paths_are_anchored_to_repo_root():
    assert chunker.ROOT_DIR == REPO_ROOT
    assert chunker.SD_DIR == REPO_ROOT / "data" / "sd"
    assert chunker.MM_DIR == REPO_ROOT / "data" / "mm"


def test_required_files_are_absolute_paths():
    assert all(p.is_absolute() for p in loader.required_files().values())


def test_no_windows_specific_paths_are_hardcoded():
    for module in (loader, chunker):
        source = Path(module.__file__).read_text(encoding="utf-8")
        assert "\\data\\" not in source
        assert "C:" not in source


def test_loader_reads_data_regardless_of_cwd(aw_dataset, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)          # a CWD unrelated to the repository
    sales = loader.load_sales_orders()
    assert {"SalesOrderID", "TotalDue", "LineTotal"} <= set(sales.columns)


def test_load_and_save_all_writes_all_clean_files(aw_dataset, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    loader.load_and_save_all()

    for name in ("sales_orders_clean.csv", "customer_master_clean.csv"):
        assert (aw_dataset / "sd" / name).exists()
    for name in ("product_master_clean.csv", "stock_levels_clean.csv",
                 "purchase_orders_clean.csv"):
        assert (aw_dataset / "mm" / name).exists()


def test_chunker_reads_clean_files_regardless_of_cwd(aw_dataset, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    loader.load_and_save_all()
    chunks = chunker.load_and_chunk_all()

    tables = {c["metadata"]["table"] for c in chunks}
    assert tables == {"sales_orders", "customer_master", "product_master",
                      "stock_levels", "purchase_orders"}


def test_chunker_reports_a_clear_error_without_clean_tables(empty_data_dir, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match="src.data.loader"):
        chunker.load_and_chunk_all()
