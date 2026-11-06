"""Preflight validation tests for src/data/loader.py.

Uses temporary fixtures only — no AdventureWorks download is required.
"""
import pytest

from src.data import loader
from src.data.loader import DatasetError, HEADERS, required_files, validate_dataset

from tests.conftest import sample_row, write_table, build_raw_dataset, point_modules_at


def test_required_files_are_the_seven_canonical_names():
    files = required_files()
    assert set(files) == set(HEADERS)
    assert len(files) == 7
    assert all(str(p).endswith(".csv") for p in files.values())


def test_missing_dataset_is_reported_with_filenames(empty_data_dir):
    with pytest.raises(DatasetError) as exc:
        validate_dataset()

    message = str(exc.value)
    for name, path in required_files().items():
        assert path.name in message, f"{name} not named in the error"
    assert "tab-delimited" in message
    assert "oltp-install-script" in message


def test_single_missing_file_is_named(empty_data_dir):
    build_raw_dataset(empty_data_dir)
    (empty_data_dir / "mm" / "ProductInventory.csv").unlink()

    with pytest.raises(DatasetError, match="ProductInventory"):
        validate_dataset()


def test_valid_fixture_dataset_passes_preflight(aw_dataset):
    table_rows = validate_dataset()
    assert all(rows > 0 for rows in table_rows.values())
    assert set(table_rows) == set(HEADERS)


def test_comma_delimited_file_is_rejected(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    build_raw_dataset(data_dir)
    point_modules_at(data_dir, monkeypatch)

    path = data_dir / "sd" / "Customer.csv"
    rows = [sample_row("Customer")]
    path.write_text(",".join(rows[0]) + "\n", encoding="utf-8")

    with pytest.raises(DatasetError, match="Customer"):
        validate_dataset()


def test_unexpected_header_row_is_rejected(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    build_raw_dataset(data_dir)
    point_modules_at(data_dir, monkeypatch)

    path = data_dir / "mm" / "Product.csv"
    write_table(path, "Product", [HEADERS["Product"], sample_row("Product")])

    with pytest.raises(DatasetError, match="header row"):
        validate_dataset()


def test_wrong_column_count_is_rejected(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    build_raw_dataset(data_dir)
    point_modules_at(data_dir, monkeypatch)

    path = data_dir / "sd" / "SalesOrderHeader.csv"
    short_row = sample_row("SalesOrderHeader")[:-1]          # one column missing
    path.write_text("\t".join(short_row) + "\n", encoding="utf-8")

    with pytest.raises(DatasetError, match=r"expected 26"):
        validate_dataset()


def test_inconsistent_column_counts_are_rejected(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    build_raw_dataset(data_dir)
    point_modules_at(data_dir, monkeypatch)

    path = data_dir / "sd" / "SalesOrderDetail.csv"
    rows = [sample_row("SalesOrderDetail"), sample_row("SalesOrderDetail") + ["extra"]]
    write_table(path, "SalesOrderDetail", rows)

    with pytest.raises(DatasetError, match="inconsistent column counts"):
        validate_dataset()


def test_empty_file_is_rejected(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    build_raw_dataset(data_dir)
    point_modules_at(data_dir, monkeypatch)

    (data_dir / "sd" / "Customer.csv").write_text("", encoding="utf-8")

    with pytest.raises(DatasetError, match="empty"):
        validate_dataset()


def test_utf8_bom_is_rejected(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    build_raw_dataset(data_dir)
    point_modules_at(data_dir, monkeypatch)

    path = data_dir / "sd" / "Customer.csv"
    path.write_text("\t".join(sample_row("Customer")) + "\n", encoding="utf-8-sig")

    with pytest.raises(DatasetError, match="BOM"):
        validate_dataset()


def test_utf16_encoding_is_rejected(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    build_raw_dataset(data_dir)
    point_modules_at(data_dir, monkeypatch)

    path = data_dir / "sd" / "Customer.csv"
    path.write_text("\t".join(sample_row("Customer")) + "\n", encoding="utf-16")

    with pytest.raises(DatasetError, match="UTF-16"):
        validate_dataset()


def test_truncated_download_warns_but_does_not_fail(tmp_path, monkeypatch, caplog):
    data_dir = tmp_path / "data"
    build_raw_dataset(data_dir)
    point_modules_at(data_dir, monkeypatch)

    with caplog.at_level("WARNING"):
        table_rows = validate_dataset()

    assert table_rows["SalesOrderHeader"] == 1
    assert "possible truncated download" in caplog.text


def test_multibyte_char_on_the_probe_boundary_is_not_a_false_encoding_fault(tmp_path, monkeypatch):
    """A valid UTF-8 file must not be rejected because the probe cut a character."""
    data_dir = tmp_path / "data"
    build_raw_dataset(data_dir)
    point_modules_at(data_dir, monkeypatch)

    path = data_dir / "sd" / "Customer.csv"
    first_line = ("\t".join(sample_row("Customer")) + "\n").encode("utf-8")
    padded = first_line + b"x" * (4095 - len(first_line))     # 4095 bytes so far
    path.write_bytes(padded + "\u00e9".encode("utf-8") + b"\n")

    problems, _ = loader._inspect_file("Customer", path)

    assert not any("UTF-8" in p or "encoding" in p for p in problems), problems


def test_read_csv_refuses_a_malformed_file(empty_data_dir):
    bad = empty_data_dir / "sd" / "Customer.csv"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_text("1,2,3\n", encoding="utf-8")

    with pytest.raises(DatasetError, match="Customer"):
        loader.read_csv(bad, "Customer")


def test_load_and_save_all_stops_before_writing(empty_data_dir):
    with pytest.raises(DatasetError):
        loader.load_and_save_all()

    assert not (empty_data_dir / "sd" / "sales_orders_clean.csv").exists()
