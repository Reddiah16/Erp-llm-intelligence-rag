import streamlit as st
import requests
import pandas as pd
from pathlib import Path

API_URL = "http://localhost:8000"

# ── Canonical data contract ──────────────────────────────────────────────
# These are the cleaned outputs written by src/data/loader.py, anchored to the
# repository root so the dashboard works from any terminal directory.
DATA_DIR   = Path(__file__).resolve().parents[1] / "data"
SALES_FILE = DATA_DIR / "sd" / "sales_orders_clean.csv"
STOCK_FILE = DATA_DIR / "mm" / "stock_levels_clean.csv"
PO_FILE    = DATA_DIR / "mm" / "purchase_orders_clean.csv"

sales = stock = po = None

st.set_page_config(
    page_title = "ERP LLM Intelligence",
    page_icon  = "🤖",
    layout     = "wide"
)

st.title("🤖 ERP LLM Intelligence — RAG Dashboard")
st.caption("Natural language querying of AdventureWorks SD and MM data")

# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.header("Quick Queries")
    st.caption("Click to run a sample query:")

    sample_queries = [
        "Which products are below safety stock level?",
        "Which products need reordering?",
        "What is the total revenue from sales orders?",
        "Which products generated the most revenue?",
        "Which purchase orders are pending?",
        "What are the most recent sales orders?",
    ]

    for q in sample_queries:
        if st.button(q, use_container_width=True):
            st.session_state["query_input"] = q

    st.markdown("---")
    if st.button("Check API health"):
        try:
            r = requests.get(f"{API_URL}/health", timeout=3)
            st.success(f"API online — {r.json()}")
        except Exception:
            st.error("API not reachable")

# ── Key metrics ───────────────────────────────────────────────────
try:
    sales = pd.read_csv(SALES_FILE)
    stock = pd.read_csv(STOCK_FILE)
    po    = pd.read_csv(PO_FILE)

    # TotalDue is ORDER-level but repeats on every joined detail line, so it is
    # counted once per SalesOrderID — never summed over the joined frame.
    orders = sales.drop_duplicates(subset=["SalesOrderID"])
    # Purchase-order status is ORDER-level too: one pending header can span
    # many detail lines, so the KPI counts orders, not lines.
    po_orders = po.drop_duplicates(subset=["PurchaseOrderID"])
    # Stock rows carry every product, but only those with a ProductInventory
    # record have a known quantity. Missing is not zero, so the stock KPIs
    # are computed over the recorded subset and never over all products.
    recorded_stock   = stock[stock["HasInventoryRecord"]]
    unrecorded_stock = stock[~stock["HasInventoryRecord"]]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Sales orders",                    f"{orders['SalesOrderID'].nunique():,}")
    col2.metric("Revenue — order-level TotalDue (USD)", f"{orders['TotalDue'].sum():,.0f}")
    col3.metric("Products below safety stock",
                f"{int(recorded_stock['BelowSafety'].sum()):,}",
                f"of {len(recorded_stock):,} products with recorded stock")
    col4.metric("Pending purchase orders",         f"{int((po_orders['Status'] == 'Pending').sum()):,}")

    st.caption(
        f"Stock KPIs cover the {len(recorded_stock):,} products that have at least "
        f"one `ProductInventory` row. {len(unrecorded_stock):,} products have no "
        "inventory record: their quantity is **unknown**, so they are excluded "
        "from the below-safety and reorder counts rather than treated as zero stock."
    )
except Exception:
    st.warning(
        "Could not load cleaned AdventureWorks data — run "
        "'python -m src.data.loader' first."
    )

st.markdown("---")

# ── Chat interface ─────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display conversation history
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Query input
query = st.chat_input(
    "Ask a question about your ERP data...",
)

# Handle sidebar quick query
if "query_input" in st.session_state and st.session_state["query_input"]:
    query = st.session_state.pop("query_input")

if query:
    # Show user message
    with st.chat_message("user"):
        st.write(query)
    st.session_state["messages"].append({"role": "user", "content": query})

    # Call API
    with st.chat_message("assistant"):
        with st.spinner("Retrieving from ERP data..."):
            try:
                resp   = requests.post(
                    f"{API_URL}/query",
                    json    = {"question": query},
                    timeout = 60
                )
                result = resp.json()
                answer  = result.get("answer", "No answer returned.")
                sources = result.get("sources", [])

                st.write(answer)
                st.session_state["messages"].append({
                    "role":    "assistant",
                    "content": answer
                })

                # Show sources
                if sources:
                    with st.expander(f"Retrieved sources ({len(sources)} records)"):
                        for s in sources:
                            st.markdown(
                                f"**{s.get('record_id', '')}** — `{s.get('table', '')}` "
                                f"(relevance: {s.get('relevance', 0):.3f})"
                            )
                            st.caption(s.get("text", ""))
                            st.divider()

                # Product-level revenue uses LINE-level LineTotal, never the
                # order-level TotalDue repeated across detail rows.
                if sales is not None and ("revenue" in query.lower() or "product" in query.lower()):
                    try:
                        top_products = (
                            sales.groupby("Name")["LineTotal"].sum()
                                 .nlargest(10)
                                 .rename("Revenue (USD, line level)")
                                 .reset_index()
                        )
                        if not top_products.empty:
                            st.bar_chart(top_products.set_index("Name"))
                    except Exception:
                        pass

                if stock is not None and any(
                        w in query.lower() for w in ("stock", "safety", "reorder")):
                    try:
                        # Only products with a recorded quantity can be short;
                        # unknown stock is not a stock-out.
                        below = stock[
                            stock["BelowSafety"] & stock["HasInventoryRecord"]
                        ][
                            ["Name", "TotalQuantity", "SafetyStockLevel", "ReorderPoint"]
                        ].head(10)
                        if not below.empty:
                            st.bar_chart(below.set_index("Name"))
                    except Exception:
                        pass

            except Exception as e:
                st.error(f"API error: {str(e)}")

# ── Clear conversation ─────────────────────────────────────────────
if st.session_state["messages"]:
    if st.button("Clear conversation"):
        st.session_state["messages"] = []
        st.rerun()