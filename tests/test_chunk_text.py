"""Missing values must never leak "nan", "None" or "NaT" into RAG chunk text.

Product attributes are frequently absent (Color, Size, Weight, ProductLine,
sell dates), and the cleaned CSVs store them as empty cells. Before the
formatting fix a single product chunk carried a dozen literal "nan" tokens,
which pollutes embeddings and invites the model to repeat them. Missing cells
must render as one readable placeholder, while genuine zeros (a price or a
quantity of 0) must survive untouched.
"""
import re

import pandas as pd
import pytest

from src.chunking.chunker import (
    MISSING_PLACEHOLDER,
    chunk_table,
    format_value,
    load_and_chunk_all,
    row_to_text,
)
from src.data import loader
from tests.conftest import sample_row, write_table

# A field value that is exactly one of these tokens means a missing value
# leaked into the text meant for the embedding model.
LEAK_PATTERN = re.compile(r"(?:: )(nan|None|NaT|<NA>|N/A)(?=(?: \||$))", re.MULTILINE)


def field_pairs(text: str) -> dict:
    """Split one rendered row back into {column: value}."""
    body = text.split("] ", 1)[1]
    pairs = {}
    for field in body.split(" | "):
        column, _, value = field.partition(": ")
        pairs[column] = value
    return pairs


# ── format_value unit behaviour ──────────────────────────────────────────
@pytest.mark.parametrize("missing", [
    float("nan"), None, pd.NaT, pd.NA,
    "nan", "NaN", "None", "NaT", "<NA>", "N/A", "n/a", "", "   ", "null",
])
def test_missing_values_become_the_placeholder(missing):
    assert format_value(missing) == MISSING_PLACEHOLDER


@pytest.mark.parametrize("present", [0, 0.0, "0", False, True, "NA", "MM",
                                    "KG", "S", "H", "US", 12.5, "Adjustable Race"])
def test_real_values_are_preserved_verbatim(present):
    """Zero and False are data, not gaps; short legitimate codes are not eaten."""
    assert format_value(present) == str(present)


def test_numeric_zero_is_not_treated_as_missing():
    row = pd.Series({"ProductID": 1, "ListPrice": 0.0, "Weight": 0,
                     "MakeFlag": False})
    pairs = field_pairs(row_to_text(row, "product_master"))

    assert pairs["ListPrice"] == "0.0"
    assert pairs["Weight"] == "0"
    assert pairs["MakeFlag"] == "False"
    assert MISSING_PLACEHOLDER not in pairs.values()


# ── missing optional product attributes ──────────────────────────────────
def test_optional_product_attributes_render_as_unknown():
    row = pd.Series({
        "ProductID": 1,
        "Name": "Adjustable Race",
        "Color": float("nan"),
        "Size": None,
        "Weight": pd.NA,
        "SellEndDate": pd.NaT,
        "ProductLine": "N/A",          # the loader's old filler value
    })
    pairs = field_pairs(row_to_text(row, "product_master"))

    for column in ("Color", "Size", "Weight", "SellEndDate", "ProductLine"):
        assert pairs[column] == MISSING_PLACEHOLDER, column
    assert pairs["Name"] == "Adjustable Race"
    assert not LEAK_PATTERN.search(row_to_text(row, "product_master"))


@pytest.fixture
def clean_tables(aw_dataset):
    """Raw fixture data with genuinely missing product attributes, then loaded.

    The default fixture fills every cell with "1", which cannot exercise the
    missing-value path, so the optional product attributes are blanked first.
    """
    write_table(aw_dataset / "mm" / "Product.csv", "Product", [
        sample_row("Product", Color="", Size="", Weight="", ProductLine="",
                   Class="", Style="", WeightUnitMeasureCode="", SellEndDate=""),
    ])
    loader.load_and_save_all()
    return aw_dataset


# ── pipeline level: no leak survives into any chunk ──────────────────────
def test_every_chunk_of_a_generated_dataset_is_leak_free(clean_tables):
    chunks = load_and_chunk_all()
    assert chunks, "fixture produced no chunks"

    leaked = [c["text"] for c in chunks if LEAK_PATTERN.search(c["text"])]
    assert not leaked, f"{len(leaked)} chunks still leak a missing-value token"

    product_chunks = [c for c in chunks if c["metadata"]["table"] == "product_master"]
    assert product_chunks
    pairs = field_pairs(product_chunks[0]["text"])
    for column in ("Color", "Size", "Weight", "ProductLine", "Class",
                   "Style", "SellEndDate"):
        assert pairs[column] == MISSING_PLACEHOLDER, column
    # Identifiers that are present must not be replaced by the placeholder.
    assert pairs["ProductID"] == "1"
    assert pairs["SafetyStockLevel"] == "1"


def test_chunking_the_real_clean_product_table_has_no_missing_value_tokens():
    """Runs only when the dataset has been loaded; guards the exact regression."""
    from src.chunking import chunker

    path = chunker.MM_DIR / "product_master_clean.csv"
    if not path.exists():
        pytest.skip("product_master_clean.csv not generated yet")

    df = pd.read_csv(path)
    assert int(df.isna().sum().sum()) > 0, "expected empty optional attributes to exist"

    chunks = chunk_table(df, "product_master", "ProductID")
    leaked = [c["text"] for c in chunks if LEAK_PATTERN.search(c["text"])]
    assert not leaked, f"real product chunks leak: {leaked[:1]}"
    assert MISSING_PLACEHOLDER in chunks[0]["text"]


def test_record_id_metadata_survives_the_formatting_change(clean_tables):
    chunks = load_and_chunk_all()
    product_chunks = [c for c in chunks if c["metadata"]["table"] == "product_master"]
    assert product_chunks[0]["metadata"]["record_id"] == "1"
