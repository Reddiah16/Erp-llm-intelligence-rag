# 🤖 ERP LLM Intelligence (RAG System)

![CI](https://github.com/kbajish/erp-llm-intelligence-rag/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Docker](https://img.shields.io/badge/docker-compose-blue)

ERP LLM Intelligence is a Retrieval-Augmented Generation (RAG) system that enables natural language querying of ERP data across Sales & Distribution (SD) and Materials Management (MM) modules. Business users can ask questions like "What are the top customers by order value?" or "Which products are below safety stock level?" and receive accurate, context-aware answers grounded in actual data rather than LLM hallucinations.

Built on the AdventureWorks dataset — a standard enterprise business database covering real sales orders, customers, products, inventory, and purchase orders. The system reduces dependency on manual reporting tools by allowing non-technical users to query ERP data using natural language.

This project demonstrates the architecture and retrieval patterns of an ERP-integrated RAG system. It is intended as a reference implementation and learning resource, not a deployment-ready system.

---

## 🚀 Key Features

- 📊 AdventureWorks dataset — real SD and MM ERP data (31,465 sales orders across 121,317 order lines, 504 products, 19,820 customers)
- 🧩 Table-aware chunking preserving column context for accurate retrieval
- 🔍 Semantic search using embeddings (nomic-embed-text via Ollama)
- 🗄 Persistent vector store using ChromaDB (no re-indexing on restart)
- 🤖 LangChain RAG pipeline with source-grounded responses
- 📚 Source citation — every answer includes retrieved records
- 📈 Retrieval evaluation using MRR and Recall@5 tracked via MLflow
- 💬 Streamlit chatbot UI with conversation memory and dynamic visualisations
- ⚡ FastAPI backend (`/query`, `/health`)
- 🐳 Docker Compose for reproducible full-stack deployment
- 🔄 GitHub Actions CI for testing and validation

---

## 🧠 Why RAG instead of SQL?

Traditional ERP reporting relies on structured queries or predefined dashboards, which require technical knowledge and limit flexibility. This system enables semantic search and natural language interaction, allowing users to ask complex, context-aware business questions without knowing database schemas or writing queries.

---

## 🧠 System Architecture

```
AdventureWorks Dataset (SD + MM modules)
        ↓
src/data/loader.py             — loads and cleans AdventureWorks CSVs
        ↓
src/chunking/chunker.py        — table-aware chunking strategy
        ↓
src/embeddings/ingest.py       — nomic-embed-text via Ollama
        ↓
chroma_db/                     — persistent ChromaDB vector store
        ↓
src/rag/chain.py               — LangChain RAG pipeline
        ↓
src/llm/prompts.py             — structured prompt templates
        ↓
api/main.py                    — FastAPI (/query, /health)
        ↓
dashboard/app.py               — Streamlit chatbot + charts
        ↓
src/evaluation/metrics.py      — MRR, Recall@5 → MLflow
```

---

## ⚙️ How It Works

AdventureWorks ERP data covering SD and MM modules is loaded, cleaned, and structured into five business tables — sales orders, customer master, product master, stock levels, and purchase orders. The loader normalises dates, maps status codes to readable labels, joins related tables (e.g. sales orders enriched with product names), and saves clean CSVs ready for chunking.

The data is processed using a table-aware chunking strategy that preserves column relationships, ensuring semantic meaning is retained during retrieval. Chunks are embedded using a local embedding model (nomic-embed-text via Ollama) and stored in a persistent ChromaDB vector database. When a user submits a query, the system retrieves the most relevant records and passes them to llama3.2 via a LangChain RAG chain. The LLM generates a response grounded in the retrieved context, along with source references showing exactly which records were used.

Retrieval quality metrics (MRR and Recall@5) are logged to MLflow after each query, enabling ongoing evaluation of the RAG pipeline.

---

## 📊 Dataset Overview

Counts refer to the canonical Microsoft `oltp-install-script` files. Watch the grain: a sales order header joins to one or more detail lines, so header counts and joined-row counts are different numbers.

| Table | Source | Rows | Grain / description |
|---|---|---|---|
| Sales orders | SalesOrderHeader + SalesOrderDetail + Product | 121,317 joined rows from 31,465 orders | One row per order line; order-level `TotalDue` repeats on each line, `LineTotal` is line-level |
| Customer master | Customer | 19,820 | Customer accounts and territories |
| Product master | Product | 504 | Products with safety stock and reorder points |
| Stock levels | ProductInventory + Product | 504 | One row per product, aggregated from 1,069 inventory rows; `HasInventoryRecord` marks the 72 products that have no inventory row and therefore no known quantity — see "Inventory coverage" |
| Purchase orders | PurchaseOrderHeader + PurchaseOrderDetail | 8,845 joined rows from 4,012 orders | One row per purchase order line, with status and quantities |

### Revenue grain — important

`TotalDue` is **order-level** and `LineTotal` is **line-level**. Because the loader joins headers onto detail lines, summing `TotalDue` over the joined frame double-counts revenue.

Measured on the full dataset, the naive sum is **USD 2,926,970,116.50** against a correct **USD 123,216,779.55** — an inflation factor of **23.75×**.

Note the factor is *not* the 121,317 ÷ 31,465 ≈ 3.86 row ratio. It equals the **value-weighted** average number of detail lines per order: high-value orders carry many more lines (up to 72), so they are inflated disproportionately. The multiplier therefore depends on how order values and line counts are distributed and will differ for other datasets — measure it, don't infer it from the row count.

Use the loader helpers instead:

```python
from src.data.loader import (
    order_level_revenue, line_level_revenue,
    purchase_order_level_total, purchase_line_total,
    pending_purchase_order_count, header_level_sum, header_level_count,
)

order_level_revenue(sales)             # TotalDue counted once per SalesOrderID
line_level_revenue(sales)              # sum of LineTotal across order lines
purchase_order_level_total(po)         # USD 70,479,332 (order grain)
purchase_line_total(po)                # sum of PO LineTotal, line grain
pending_purchase_order_count(po)       # 225 pending ORDERS, not 477 detail lines

# Generic form for any header-level column:
header_level_sum(sales, "SalesOrderID", "Freight")
header_level_count(po, "PurchaseOrderID", Status="Approved")
```

The same double-count applies to purchase orders: 8,845 joined rows over 4,012 orders inflate `TotalDue` from USD 70,479,332 to USD 180,741,103, and counting joined rows labelled `Pending` reports 477 instead of 225 orders. This separation is enforced by `tests/test_revenue_grain.py` and `tests/test_dashboard_contract.py`.

### Inventory coverage — three distinct states

`stock_levels_clean.csv` contains **one row per product, all 504**, because the join is driven by `Product` rather than by the inventory aggregation. A missing `ProductInventory` row is **not** a zero quantity, so `HasInventoryRecord` keeps the three states apart:

| State | Evidence in the data | `TotalQuantity` | `HasInventoryRecord` | Flags | Correct reading |
|---|---|---|---|---|---|
| Known inventory quantity | inventory rows with a non-zero sum | real figure | `True` | computed | genuine on-hand level |
| Confirmed zero quantity | inventory rows exist, quantities sum to `0` | `0` | `True` | `True` | genuinely stocked out; legitimately below safety stock |
| Missing inventory record | no `ProductInventory` rows at all | **empty** | `False` | `False` | stock level **unknown** — neither healthy nor stocked out |

Measured on the canonical dataset: **432 products have an inventory record** (4 of them at a confirmed quantity of zero) and **72 have none**. All 72 are manufactured finished goods (`MakeFlag = 1`, `FinishedGoodsFlag = 1`) with a positive `SafetyStockLevel`, 56 of them appear in `SalesOrderDetail`, and none appear in `PurchaseOrderDetail` — consistent with make-to-order frames and bicycles that AdventureWorks simply does not stock. Filling their quantity with zero would silently invent 72 "below safety stock" alerts, so the quantity is left missing instead.

Consequences of the wider table, all verified against the previous output: the 432 previously reported rows keep identical quantities, safety levels, reorder points and flags, and the alert counts do not move — **69 below safety stock, 7 needing reorder**. Chunking gains 72 `stock_levels` chunks (still far under the 2,000-row sampling limit), and unknown quantities render as `TotalQuantity: Unknown`, never `nan`.

Count stock over the recorded subset, never over the whole table:

```python
from src.data.loader import (
    recorded_stock_product_count,   # 432 — the honest KPI denominator
    below_safety_product_count,     # 69  — recorded stock only
    needs_reorder_product_count,    # 7   — recorded stock only
)

stock = pd.read_csv("data/mm/stock_levels_clean.csv")   # 504 rows
recorded = stock[stock["HasInventoryRecord"]]           # 432 rows
```

The dashboard applies the same split in its "Products below safety stock" KPI and states the distinction under the metric row. `tests/test_inventory_coverage.py` and `tests/test_dashboard_contract.py` enforce it.

---

## 📊 Dashboard Overview

The Streamlit dashboard provides:

- 💬 Chat interface for natural language ERP queries
- 📊 Dynamic bar charts for numeric responses
- 📚 Retrieved source records panel for answer transparency
- 🔢 Key business metrics — sales order count, order-level revenue (USD), products below safety stock (counted over products with a recorded inventory quantity only), pending purchase orders
- 🔁 Conversation history within the session

---

## 🛠 Tech Stack

| Layer | Tool |
|---|---|
| Dataset | AdventureWorks (Microsoft open dataset) |
| Embeddings | nomic-embed-text (Ollama) |
| Vector store | ChromaDB (persistent, local) |
| RAG orchestration | LangChain |
| LLM | llama3.2 (Ollama, local) |
| Backend | FastAPI, Uvicorn |
| Dashboard | Streamlit |
| Experiment tracking | MLflow |
| Containerisation | Docker Compose |
| CI/CD | GitHub Actions |
| Testing | pytest |

---

## 📂 Project Structure

```
erp-llm-intelligence-rag/
│
├── data/                          # ERP CSV files (not committed)
│   ├── sd/                        # Sales & Distribution tables
│   │   ├── SalesOrderHeader.csv   # Raw AdventureWorks files
│   │   ├── SalesOrderDetail.csv
│   │   ├── Customer.csv
│   │   ├── sales_orders_clean.csv # Cleaned output from loader
│   │   └── customer_master_clean.csv
│   └── mm/                        # Materials Management tables
│       ├── Product.csv            # Raw AdventureWorks files
│       ├── ProductInventory.csv
│       ├── PurchaseOrderHeader.csv
│       ├── PurchaseOrderDetail.csv
│       ├── product_master_clean.csv  # Cleaned output from loader
│       ├── stock_levels_clean.csv
│       └── purchase_orders_clean.csv
│
├── chroma_db/                     # ChromaDB persistent store (not committed)
│
├── src/
│   ├── data/
│   │   └── loader.py              # AdventureWorks data loader + cleaner
│   ├── chunking/
│   │   └── chunker.py             # Table-aware chunking logic
│   ├── embeddings/
│   │   └── ingest.py              # Embedding + ChromaDB ingestion
│   ├── rag/
│   │   └── chain.py               # LangChain RAG chain
│   ├── llm/
│   │   └── prompts.py             # Prompt templates
│   └── evaluation/
│       └── metrics.py             # MRR, Recall@5, MLflow logging
│
├── api/
│   └── main.py                    # FastAPI app
│
├── dashboard/
│   └── app.py                     # Streamlit chatbot UI
│
├── tests/
│   ├── conftest.py                # Tiny tab-delimited fixtures (no download needed)
│   ├── test_imports.py            # CI-safe import tests
│   ├── test_loader.py             # Data loader smoke tests (skip without real data)
│   ├── test_loader_preflight.py   # Missing files, delimiters, headers, column counts, encoding
│   ├── test_data_paths.py         # Repository-root path anchoring and CWD independence
│   ├── test_revenue_grain.py      # Sales and purchase-order grain regression
│   ├── test_inventory_coverage.py # All 504 products listed; unknown stock never counted as zero
│   ├── test_chunk_text.py         # No nan/None/NaT leaking into RAG chunk text
│   └── test_dashboard_contract.py # Dashboard/API consume the canonical loader schema
│
├── .github/
│   └── workflows/
│       └── ci.yml                 # GitHub Actions pipeline
│
├── mlruns/                        # MLflow tracking (not committed)
├── docker-compose.yml
├── Dockerfile.api
├── Dockerfile.dashboard
├── .env.example
├── requirements.api.txt
├── requirements.dashboard.txt
├── requirements.dev.txt
├── AI_ACT_COMPLIANCE.md
└── README.md
```

---

## ▶️ Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/kbajish/erp-llm-intelligence-rag.git
cd erp-llm-intelligence-rag
```

### 2. Create and activate virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Linux/Mac
```

### 3. Download the required dataset

The canonical dataset is Microsoft's AdventureWorks `oltp-install-script` set — [download here](https://github.com/Microsoft/sql-server-samples/tree/master/samples/databases/adventure-works/oltp-install-script). Seven files are required, in this exact layout:

```
data/sd/  → SalesOrderHeader.csv, SalesOrderDetail.csv, Customer.csv
data/mm/  → Product.csv, ProductInventory.csv, PurchaseOrderHeader.csv, PurchaseOrderDetail.csv
```

**Format requirements — read carefully:**

| Requirement | Expected | Not accepted |
|---|---|---|
| Delimiter | TAB (`\t`) | comma-separated CSV |
| Header row | none (headerless) | files whose first line is column names |
| Encoding | UTF-8 / ANSI, no BOM | UTF-8 BOM, UTF-16 |
| Column count per file | 26 / 11 / 7 / 25 / 7 / 13 / 11 | any other count |
| Variant | OLTP `oltp-install-script` | AdventureWorksDW or Excel/SSMS re-exports |

Column names are supplied by `HEADERS` in `src/data/loader.py`, which is why the files must not carry a header row. Filenames are case-sensitive; `Product.csv` is read by four of the five loaders, so a single missing file breaks several tables.

### 4. Load and clean the data

```bash
python -m src.data.loader
```

The loader runs a **preflight validation** over all seven files first, reporting any missing file, wrong delimiter, unexpected header row, wrong column count or encoding problem before touching the data. Paths are anchored to the repository root, so this works from any directory. Cleaned outputs are written next to the raw files as `*_clean.csv`.

> Retrieval preconditions can be checked independently (row and column counts per file, no writes) with `python -c "from src.data.loader import validate_dataset; validate_dataset()"`.

`src/data/generator.py` produces a small synthetic SAP-style dataset for offline experimentation only. It is **not** part of this workflow, and the dashboard, API and RAG chain do not depend on it.

### 5. Ingest data into ChromaDB
```bash
python -m src.embeddings.ingest
```

### 6. Start all services
```bash
docker compose up --build
```

### 7. Access services

| Service   | URL                        |
|-----------|----------------------------|
| API       | http://localhost:8000      |
| API docs  | http://localhost:8000/docs |
| Dashboard | http://localhost:8501      |
| MLflow    | http://localhost:5000      |

---

## 🧪 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/query` | Natural language query — returns answer, sources, retrieval scores |
| `GET`  | `/health` | Health check |

---

## 🧪 Testing

The suite runs on temporary tab-delimited fixtures, so no dataset download is required:

```bash
pip install -r requirements.dev.txt
pytest tests/ -v                       # full suite
pytest tests/test_loader_preflight.py  # preflight validation only
pytest tests/test_revenue_grain.py     # order vs line grain guard (sales + POs)
pytest tests/test_inventory_coverage.py # inventory coverage and unknown-not-zero
pytest tests/test_chunk_text.py        # missing values render as "Unknown"
```

`tests/test_loader.py` data-dependent tests skip automatically when the real AdventureWorks files are absent.

---

## 📈 Possible Extensions

- Hybrid retrieval (BM25 + dense embeddings) for better recall
- Fine-tuned embedding model on ERP-specific terminology
- Role-based access control (RBAC) on the API
- Integration with real SAP systems via OData APIs
- Cloud deployment (AWS/GCP) with managed ChromaDB

---

## 👤 Author

Experienced IT professional with a background in development, cybersecurity, and ERP systems, with expertise in Industrial AI. Focused on building well-engineered AI systems with explainability, LLM integration, and MLOps practices.
