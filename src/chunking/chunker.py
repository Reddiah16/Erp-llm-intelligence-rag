import pandas as pd
from pathlib import Path
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")
log = logging.getLogger(__name__)

# Anchored to the repository root, matching src/data/loader.py, so chunking
# works from any terminal directory.
ROOT_DIR = Path(__file__).resolve().parents[2]
SD_DIR = ROOT_DIR / "data" / "sd"
MM_DIR = ROOT_DIR / "data" / "mm"


# Missing cells are rendered with one readable placeholder instead of leaking
# "nan", "None", "NaT" or the loader's "N/A" filler into embedded text.
MISSING_PLACEHOLDER = "Unknown"

# Compared case-insensitively against the WHOLE cell, so legitimate short
# codes are never mistaken for missing data. Bare "na" is deliberately
# excluded from the set for exactly that reason.
MISSING_SENTINELS = {"", "nan", "none", "nat", "null", "n/a", "<na>", "<null>"}


def format_value(val) -> str:
    """Render one cell for chunk text, mapping missing values to a placeholder.

    Zero is a meaningful quantity or price and is never treated as missing;
    only absent values (NaN, NaT, None, pd.NA) and the sentinel strings are.
    """
    try:
        if pd.isna(val):
            return MISSING_PLACEHOLDER
    except (TypeError, ValueError):
        pass  # non-scalar cells fall through to their textual form

    text = str(val)
    if text.strip().lower() in MISSING_SENTINELS:
        return MISSING_PLACEHOLDER
    return text


def row_to_text(row: pd.Series, table_name: str) -> str:
    fields = " | ".join([f"{col}: {format_value(val)}" for col, val in row.items()])
    return f"[{table_name}] {fields}"


def chunk_table(df: pd.DataFrame, table_name: str,
                id_col: str = None) -> List[Dict]:
    chunks = []
    for _, row in df.iterrows():
        text     = row_to_text(row, table_name)
        metadata = {"table": table_name, "source": f"{table_name} record"}
        if id_col and id_col in row:
            metadata["record_id"] = str(row[id_col])
        chunks.append({"text": text, "metadata": metadata})
    return chunks


def load_and_chunk_all() -> List[Dict]:
    log.info("Loading and chunking AdventureWorks tables...")
    all_chunks = []

    tables = {
        "sales_orders":    (SD_DIR / "sales_orders_clean.csv",    "SalesOrderID"),
        "customer_master": (SD_DIR / "customer_master_clean.csv", "CustomerID"),
        "product_master":  (MM_DIR / "product_master_clean.csv",  "ProductID"),
        "stock_levels":    (MM_DIR / "stock_levels_clean.csv",    "ProductID"),
        "purchase_orders": (MM_DIR / "purchase_orders_clean.csv", "PurchaseOrderID"),
    }

    for table_name, (path, id_col) in tables.items():
        if not path.exists():
            log.warning(f"File not found: {path}")
            continue

        df = pd.read_csv(path)

        # Sample large tables to keep embedding manageable
        if len(df) > 2000:
            log.info(f"  Sampling {table_name} from {len(df)} to 2000 rows")
            df = df.sample(2000, random_state=42).reset_index(drop=True)

        chunks = chunk_table(df, table_name, id_col)
        all_chunks.extend(chunks)
        log.info(f"  {table_name}: {len(chunks)} chunks")

    if not all_chunks:
        raise FileNotFoundError(
            f"No cleaned tables found in {SD_DIR} / {MM_DIR}. "
            "Run 'python -m src.data.loader' first to produce the *_clean.csv files."
        )

    log.info(f"Total chunks: {len(all_chunks)}")
    return all_chunks


if __name__ == "__main__":
    chunks = load_and_chunk_all()
    print("\nSample chunk:")
    print(chunks[0]["text"])
    print("\nMetadata:", chunks[0]["metadata"])