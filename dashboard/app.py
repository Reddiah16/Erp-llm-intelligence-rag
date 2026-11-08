"""ERP Business Intelligence dashboard: the Streamlit face of the API.

This page owns presentation and HTTP calls only. Every figure comes from the
FastAPI backend - /analytics for the approved queries, /chat for the
assistant, /documents/upload for the document store and /history for the
session transcripts. No business KPI is recomputed here and nothing is sent
to a language model from this file: numbers in charts and tables are drawn
exactly from what the API returned.

Run with:
    .\\.venv\\Scripts\\python.exe -m streamlit run dashboard/app.py
"""
import os
import uuid
from decimal import Decimal
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

from src.evaluation.dashboard_views import route_accuracy_frame

# The API location is configuration, not a constant buried in code: the API
# may run on another host or port.
API_URL = os.getenv("ERP_API_URL") or "http://localhost:8000"

CHAT_TIMEOUT_SECONDS = 240      # a hybrid question can generate for a while
DEFAULT_TIMEOUT_SECONDS = 60
MAX_DOCUMENT_UPLOAD_BYTES = 2_000_000

# ── Canonical data contract ──────────────────────────────────────────────
# These are the cleaned outputs written by src/data/loader.py, anchored to the
# repository root so the dashboard works from any terminal directory.
DATA_DIR   = Path(__file__).resolve().parents[1] / "data"
SALES_FILE = DATA_DIR / "sd" / "sales_orders_clean.csv"
STOCK_FILE = DATA_DIR / "mm" / "stock_levels_clean.csv"
PO_FILE    = DATA_DIR / "mm" / "purchase_orders_clean.csv"

PAGES = ("Overview", "AI Assistant", "Business Analytics",
         "Evaluation & Performance", "Documents", "History")

# The approved queries rendered as ranked lists (a table plus a chart), and
# the widget each singleton query gets. The query definitions themselves stay
# in src/ai/sql_tools.py - this is only a presentation choice per name.
LIST_QUERIES = ("top_products_by_revenue", "top_customers_by_revenue",
                "products_below_safety_stock", "products_needing_reorder")

# The anonymous pilot identities and where their feedback rate line lives.
TESTER_OPTIONS = ("(not set)",) + tuple(f"Tester {index:02d}"
                                        for index in range(1, 11))

ROUTE_LABELS = {
    "sql": "SQL - business figures",
    "rag": "Policy documents",
    "hybrid": "Hybrid - figures and policy",
    "unsupported": "No approved source",
}

st.set_page_config(
    page_title = "ERP LLM Intelligence",
    page_icon  = "🤖",
    layout     = "wide"
)

st.title("🤖 ERP Business Intelligence Assistant")
st.caption("Natural language and approved analytics over the ERP data, "
           "served entirely by the FastAPI backend")


# ── API access ───────────────────────────────────────────────────────────
def call_api(method: str, path: str, timeout: int = DEFAULT_TIMEOUT_SECONDS,
             **kwargs):
    """One HTTP call, or a readable failure for the page to show.

    A network fault and a 4xx/5xx are both returned as (ok, message) rather
    than raised: a dashboard that dies on an unreachable backend shows less
    than one that says which backend it could not reach.
    """
    url = f"{API_URL}{path}"
    try:
        response = requests.request(method, url, timeout=timeout, **kwargs)
    except requests.RequestException as exc:
        return False, (f"the API at {API_URL} is not reachable "
                       f"({type(exc).__name__}). Start it with: uvicorn "
                       f"api.main:app --port 8000")
    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        return False, f"the API responded with {response.status_code}: {detail}"
    try:
        return True, response.json()
    except ValueError:
        return False, "the API returned a response that is not JSON"


def ensure_session() -> str:
    """A conversation id that survives reruns and page switches.

    Sent with every /chat call so the API keeps one transcript per
    conversation; the History page offers other sessions for browsing.
    """
    if st.session_state.get("session_id") is None:
        st.session_state["session_id"] = uuid.uuid4().hex   # 32 hex chars
    return st.session_state["session_id"]


def money(value) -> str:
    """Thousand separators for display, without touching the decimals.

    The API hands back exact database figure strings; grouping the integer
    part keeps every digit of the original value on screen.
    """
    text = str(value).strip()
    negative = text.startswith("-")
    digits = text[1:] if negative else text
    integer, _, decimals = digits.partition(".")
    try:
        grouped = f"{int(integer):,}"
    except ValueError:
        return text
    grouped = grouped + (("." + decimals) if decimals else "")
    return f"-{grouped}" if negative else grouped


def chart_values(rows, value_column: str, name_column: str):
    """Two columns for a chart: names and floating-point plot values.

    The chart is a viewport rendering, so floats are acceptable there; the
    table shown beside every chart keeps the exact API strings.
    """
    frame = pd.DataFrame(rows)
    if value_column not in frame.columns or name_column not in frame.columns:
        return None
    data = pd.DataFrame({
        name_column: [row.get(name_column, "") for row in rows],
        value_column: [float(row.get(value_column) or 0) for row in rows],
    }).set_index(name_column)
    return data


def query_table(query: dict) -> None:
    """One /analytics query as caption, exact table, optional chart."""
    st.subheader(query["title"])
    st.caption(f"grain: {query['grain']} — {query['grain_note']}")
    if not query["rows"]:
        st.info("The query ran and returned no rows. There is no figure to "
                "report - nothing is assumed to be zero.")
        return
    frame = pd.DataFrame(query["rows"])
    st.dataframe(frame, use_container_width=True, hide_index=True)
    if query.get("truncated"):
        st.caption("The API returned more rows than it shows here; "
                   f"{query['row_count']} rows exist in total.")


# ── Overview ────────────────────────────────────────────────────────────
def page_overview() -> None:
    st.header("Overview")
    st.caption("Service status and live database KPIs when the API is available; "
               "local snapshot values are labeled when used.")

    ok, response = call_api("GET", "/health")
    if not ok:
        st.error(str(response))
    else:
        components = response["components"]
        database = components["database"]
        model = components["llm"]
        docs = components["documents"]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("API", f"v{response['version']}", "reachable")
        col2.metric("PostgreSQL",
                    "online" if database["available"] else "offline",
                    database.get("detail", "")[:60])
        col3.metric("Language model",
                    "ready" if model["available"] else "unavailable",
                    model.get("model", ""))
        col4.metric("Policy documents",
                    f"{docs.get('chunks', 0)} chunks" if docs.get("available")
                    else "not ingested",
                    f"{docs.get('documents', 0)} files")
        if not database["available"] or not model["available"]:
            st.warning("Some components are unavailable - the assistant "
                       "degrades honestly rather than inventing figures. "
                       "Approved analytics still work whenever PostgreSQL "
                       "is online.")
        with st.expander("Component details"):
            st.json(response["components"])

    st.markdown("---")
    render_local_snapshot()


def render_local_snapshot() -> None:
    """The Phase 1 contract: cleaned loader files, KPIs computed correctly.

    Kept exactly as it was: order-grain totals, order-grain pending counts
    and stock KPIs over recorded inventory only."""
    analytics_ok, analytics = call_api(
        "GET", "/analytics",
        params={"question": "total_sales_revenue", "limit": 1})
    live_revenue = None
    if analytics_ok:
        results = analytics.get("queries", [])
        if results and results[0].get("rows"):
            live_revenue = results[0]["rows"][0].get("order_total_due")

    sales = stock = po = None
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
        local_revenue = sum(
            (Decimal(str(value)) for value in orders["TotalDue"]),
            Decimal("0"))

        with st.expander("Local data snapshot (Phase 1 cleaned files)",
                         expanded=True):
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Sales orders",
                        f"{orders['SalesOrderID'].nunique():,}")
            if live_revenue is not None:
                col2.metric("Revenue — order-level TotalDue (USD)",
                            money(live_revenue), "live PostgreSQL")
            else:
                col2.metric("Revenue — local snapshot TotalDue (USD)",
                            money(local_revenue), "not live PostgreSQL")
            col3.metric("Products below safety level",
                        f"{int(recorded_stock['BelowSafety'].sum()):,}",
                        f"of {len(recorded_stock):,} products with "
                        "recorded stock")
            col4.metric("Pending purchase orders",
                        f"{int((po_orders['Status'] == 'Pending').sum()):,}")
            if live_revenue is None:
                st.caption("Revenue is from the cleaned CSV snapshot because "
                           "the live analytics query was unavailable.")

            st.caption(
                f"Stock KPIs cover the {len(recorded_stock):,} products that "
                f"have at least one `ProductInventory` row. "
                f"{len(unrecorded_stock):,} products have no inventory "
                "record: their quantity is **unknown**, so they are excluded "
                "from the below-safety and reordering counts rather than "
                "treated as zero stock.")
    except Exception:
        if live_revenue is not None:
            st.metric("Revenue — order-level TotalDue (USD)",
                      money(live_revenue), "live PostgreSQL")
        st.caption(
            "Could not load cleaned AdventureWorks data — run "
            "'python -m src.data.loader' first. The API-based analytics page "
            "does not need it.")


# ── AI Assistant ────────────────────────────────────────────────────────
def page_assistant() -> None:
    st.header("AI Assistant")
    st.caption("Questions are classified by the backend: figures come from "
               "approved SQL, policy wording from the document corpus, and "
               "every stated figure is checked against the evidence.")
    tester = st.session_state.get("pilot_tester") or "(not set)"
    st.caption(f"Anonymous tester for feedback: {tester}")

    # Quick queries, consistent with the earlier dashboard behaviour.
    sample_rows = (st.columns(3), st.columns(3))
    slots = [column for row in sample_rows for column in row]
    sample_queries = [
        "What is the total revenue from sales orders?",
        "Which products are below safety stock level?",
        "How many purchase orders are pending?",
        "Which products generated the most revenue?",
        "Which customers generated the most revenue?",
        "What is the rule for products below safety stock?",
    ]
    for slot, question in zip(slots, sample_queries):
        if slot.button(question, use_container_width=True):
            st.session_state["pending_question"] = question

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    for turn_index, record in enumerate(st.session_state["messages"]):
        with st.chat_message(record["role"]):
            if record["role"] == "user":
                st.write(record["content"])
            elif record["payload"] is None:
                st.error(record.get("error", "the backend returned an error"))
            else:
                render_answer(record["payload"])
                feedback_controls(record["payload"], turn_index)

    submitted = st.chat_input("Ask a question about your ERP business...")
    pending = st.session_state.pop("pending_question", None)
    question = submitted or pending
    if question:
        st.session_state["messages"].append({"role": "user",
                                             "content": question})
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            with st.spinner("Asking the backend (classify, retrieve, "
                            "verify)..."):
                tester = st.session_state.get("pilot_tester") or ""
                ok, response = call_api(
                    "POST", "/chat", timeout=CHAT_TIMEOUT_SECONDS,
                    json={"question": question,
                          "session_id": ensure_session(),
                          "tester_id": tester if tester != "(not set)" else ""})
            if not ok:
                st.error(str(response))
                st.session_state["messages"].append({
                    "role": "assistant", "payload": None,
                    "error": str(response)})
            else:
                render_answer(response)
                st.session_state["messages"].append({
                    "role": "assistant", "payload": response, "error": None})
        st.rerun()

    if st.session_state["messages"]:
        if st.button("Clear conversation display"):
            st.session_state["messages"] = []
            st.rerun()


def feedback_controls(payload: dict, index: int) -> None:
    """Rating widgets for one answer, wired to POST /feedback.

    A grade can only be submitted against the turn_id the API returned, so a
    rating always lands on the exact exchange it grades. The written comment
    is optional and the interface asks for no personal details.
    """
    turn_id = payload.get("turn_id") or ""
    label = f"turn {index}" if not turn_id else f"turn {turn_id[:8]}…"
    with st.expander(f"Rate this answer ({label}) — pilot evaluation"):
        if not turn_id:
            st.info("This exchange was answered before evaluation recording "
                    "was active, so it has no turn id to grade.")
            return
        helpful = st.radio("Is this answer helpful?", ("Yes", "No"),
                           index=None, horizontal=True,
                           key=f"fb-help-{turn_id}-{index}")
        correct = st.radio("Is the answer factually correct?",
                           ("Yes", "No", "Not sure"), index=None,
                           key=f"fb-correct-{turn_id}-{index}")
        citations = st.radio("Are the citations relevant?",
                             ("Yes", "No", "Not applicable"), index=None,
                             key=f"fb-cites-{turn_id}-{index}")
        comment = st.text_area(
            "Optional written feedback (no personal details please)",
            key=f"fb-comment-{turn_id}-{index}")
        if st.button("Submit feedback", key=f"fb-send-{turn_id}-{index}"):
            body: dict = {
                "turn_id": turn_id,
                "tester_id": st.session_state.get("pilot_tester") or "",
            }
            if helpful is not None:
                body["helpful"] = helpful == "Yes"
            if correct is not None:
                body["correct"] = {"Yes": "yes", "No": "no",
                                   "Not sure": "not_sure"}[correct]
            if citations is not None:
                body["citations_relevant"] = {
                    "Yes": "yes", "No": "no",
                    "Not applicable": "not_applicable"}[citations]
            if comment.strip():
                body["comment"] = comment.strip()
            ok, result = call_api("POST", "/feedback", json=body)
            if ok and result.get("recorded"):
                st.success(f"Feedback recorded for question "
                           f"`{result.get('question_id')}` — thank you.")
            else:
                st.error(str(result))


def render_answer(payload: dict) -> None:
    """One /chat response with every guarantee the API reported."""
    st.caption(f"Route: {ROUTE_LABELS.get(payload['route'], payload['route'])}"
               f" · classifier: {payload['classifier']} · model: "
               f"{payload['model']}")
    st.markdown(payload["answer"])

    if payload["unverified_figures"]:
        st.error("Figures the model stated that are NOT in the evidence: "
                 f"{', '.join(payload['unverified_figures'])}")
    for warning in payload["warnings"]:
        st.warning(warning)
    if not payload["data_available"] and payload["query"]:
        st.caption("The structured figure was unavailable; nothing was "
                   "invented in its place.")
    if not payload["documents_available"] and payload["route"] in ("rag",
                                                                   "hybrid"):
        st.caption("The policy corpus was unreachable for this question.")

    sql = payload.get("sql")
    if sql:
        with st.expander(f"Evidence: {sql['title']} "
                         f"({sql['row_count']} rows, {sql['grain']} grain)"):
            st.caption(sql["grain_note"])
            st.dataframe(pd.DataFrame(sql["rows"]), use_container_width=True,
                         hide_index=True)
    citations = payload.get("citations") or []
    if citations:
        with st.expander(f"Document citations ({len(citations)})"):
            for citation in citations:
                st.markdown(
                    f"**{citation['id']} · {citation['document']}** — "
                    f"{citation['section']} "
                    f"(relevance {citation.get('relevance', 0):.2f})")
                st.caption(citation.get("excerpt", ""))
                st.divider()


# ── Business Analytics ──────────────────────────────────────────────────
def page_analytics() -> None:
    st.header("Business Analytics")
    st.caption("Figures come straight from the approved SQL the router is "
               "held to - no model writes a number here, so every digit on "
               "this page is database output.")

    limit = st.selectbox("Rows per ranked list", (5, 10, 25, 50), index=1)
    if st.button("Refresh figures"):
        st.rerun()

    ok, response = call_api("GET", "/analytics", params={"limit": limit})
    if not ok:
        st.error(str(response))
        return

    by_name = {query["query"]: query for query in response["queries"]}
    if response.get("unavailable"):
        for item in response["unavailable"]:
            st.warning(f"{item['query']} could not run: {item['detail']}")

    revenue = by_name.get("total_sales_revenue")
    if revenue and revenue["rows"]:
        row = revenue["rows"][0]
        st.subheader("Total sales and order count")
        st.caption(f"{revenue['grain']} — {revenue['grain_note']}")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Sales orders", f"{int(row['orders']):,}")
        col2.metric("Order-level total due (USD)",
                    money(row["order_total_due"]),
                    "sum of TotalDue per order header")
        col3.metric("Line-level revenue (USD)", money(row["line_revenue"]),
                    f"{int(row['order_lines']):,} order lines")
        col4.metric("Units sold", f"{int(row['units_sold']):,}")

    trends = by_name.get("monthly_sales_trends")
    if trends:
        st.subheader("Monthly sales trends")
        st.caption(f"{trends['grain']} — the line chart uses line-level "
                   "revenue; the exact order-level figures are in the table "
                   "below.")
        if trends["rows"]:
            index = [str(row["month"]) for row in trends["rows"]]
            line_series = [float(row.get("line_revenue") or 0)
                           for row in trends["rows"]]
            order_series = [float(row.get("order_total_due") or 0)
                            for row in trends["rows"]]
            frame = pd.DataFrame({
                "line revenue (line grain)": line_series,
                "order total due (order grain)": order_series,
            }, index=pd.to_datetime(index))
            st.line_chart(frame)
            with st.expander("Exact monthly figures"):
                query_table(trends)
        else:
            st.info("No monthly rows were returned.")

    for name in LIST_QUERIES:
        query = by_name.get(name)
        if not query:
            continue
        st.markdown("---")
        query_table(query)
        if name == "top_products_by_revenue":
            data = chart_values(query["rows"], "revenue", "name")
            if data is not None:
                st.bar_chart(data)
        elif name == "top_customers_by_revenue":
            data = chart_values(query["rows"], "revenue", "account_number")
            if data is not None:
                st.bar_chart(data)

    st.markdown("---")
    pending = by_name.get("pending_purchase_order_value")
    count = by_name.get("pending_purchase_order_count")
    st.subheader("Pending purchase orders")
    if pending and pending["rows"]:
        row = pending["rows"][0]
        st.caption(f"{pending['grain']} — {pending['grain_note']}")
        col1, col2, col3 = st.columns(3)
        col1.metric("Pending orders", f"{int(row['pending_orders']):,}")
        col2.metric("Pending merchandise (USD)",
                    money(row.get("pending_merchandise")))
        col3.metric("Pending total due (USD)",
                    money(row.get("pending_total_due")))
    elif count and count["rows"]:
        st.metric("Pending orders",
                  f"{int(count['rows'][0]['pending_orders']):,}")
    else:
        st.info("The pending purchase order figures are not available right "
                "now.")


# ── Evaluation & Performance ────────────────────────────────────────────
def page_evaluation() -> None:
    st.header("Evaluation & Performance")
    st.caption("Measured on all recorded assistant interactions "
               "(src/evaluation/pilot.py). Provenance is not consistently "
               "recorded, so these metrics cannot be attributed to internal "
               "QA, manual testing, or an external pilot. Tester labels are "
               "anonymous and do not verify participant type. Every "
               "percentage shows its sample size; with no data the page says "
               "so rather than inventing a figure.")

    st.subheader("Pilot tester (anonymous)")
    stored_tester = st.session_state.get("pilot_tester") or "(not set)"
    if stored_tester not in TESTER_OPTIONS:
        stored_tester = "(not set)"
    if "pilot_tester_selection" not in st.session_state:
        st.session_state["pilot_tester_selection"] = stored_tester
    tester = st.selectbox(
        "Who is using the app right now?", TESTER_OPTIONS,
        key="pilot_tester_selection", label_visibility="collapsed")
    st.session_state["pilot_tester"] = "" if tester == "(not set)" else tester
    st.caption("Tester ids are anonymous labels - they are never resolved to "
               "a person.")

    ok, response = call_api("GET", "/evaluation/metrics")
    if not ok:
        st.error(str(response))
        return
    metrics = response

    st.markdown("---")
    st.subheader("Headline metrics")
    def percent_block(column, title, payload: dict):
        value, sample = payload.get("value"), payload.get("sample", 0)
        if value is None:
            column.metric(title, "Not enough data", f"sample {sample}")
        else:
            column.metric(title, f"{value:.1f}%", f"sample {sample}")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total questions evaluated", f"{metrics['total_questions']:,}",
                f"{metrics['evaluated_turns']:,} with feedback")
    percent_block(col2, "Answer accuracy", metrics["answer_accuracy"])
    percent_block(col3, "SQL execution success",
                  metrics["sql_execution_success"])
    latency = metrics["average_latency_ms"]
    if latency["value"] is None:
        col4.metric("Avg response latency", "Not enough data",
                    f"sample {latency['sample']}")
    else:
        col4.metric("Avg response latency",
                    f"{latency['value']:,.0f} ms", f"sample {latency['sample']}")

    col5, col6, col7, col8 = st.columns(4)
    percent_block(col5, "Retrieval accuracy", metrics["retrieval_accuracy"])
    percent_block(col6, "Data grounding", metrics["data_grounding"])
    cost = metrics["cost_per_query"]
    if cost["value"] is None:
        col7.metric("Cost per query", "Not enough data"
                    if metrics["total_questions"] else "No queries yet",
                    f"{cost.get('sample', 0)} known · "
                    f"{cost.get('unknown_count', 0)} unknown")
    else:
        col7.metric("Cost per query", f"₹{cost['value']:.2f}",
                    f"{cost.get('sample', 0)} known · "
                    f"{cost.get('unknown_count', 0)} unknown")
    col7.caption(cost.get("note", "API cost only; infrastructure is not included"))
    percent_block(col8, "Task completion", metrics["task_completion"])

    st.markdown("---")
    left, right = st.columns(2)
    with left:
        st.subheader("Answer accuracy by route")
        routes = metrics["accuracy_by_route"]
        chart_frame = route_accuracy_frame(routes)
        st.bar_chart(chart_frame)
        rows = [{"route": route,
                 "accuracy": (f"{stats['value']:.1f}%"
                              if stats["value"] is not None else "Not enough data"),
                 "sample": stats["sample"]}
                for route, stats in routes.items()]
        st.dataframe(pd.DataFrame(rows), use_container_width=True,
                     hide_index=True)
    with right:
        st.subheader("Response latency over time")
        series = metrics.get("latency_series") or []
        if series:
            frame = pd.DataFrame(
                {"latency (ms)": [point["ms"] for point in series]},
                index=pd.to_datetime([point["ts"] for point in series]))
            st.line_chart(frame)
        else:
            st.info("Not enough data - no timed responses recorded yet.")

    left2, right2 = st.columns(2)
    with left2:
        st.subheader("Positive vs negative feedback")
        feedback = metrics["feedback"]
        bar = pd.DataFrame({
            "helpful yes": [feedback["helpful_yes"]],
            "helpful no": [feedback["helpful_no"]],
        })
        st.bar_chart(bar.set_index("helpful yes"))
        st.caption(f"{feedback['turns']} turns rated · "
                   f"{feedback['correct_not_sure']} 'not sure' on correctness · "
                   f"{feedback['comments']} written comments")
    with right2:
        st.subheader("Evaluation results over time")
        over_time = metrics.get("over_time") or []
        if over_time:
            frame = pd.DataFrame({
                "questions evaluated": [day["evaluated"] for day in over_time],
                "marked helpful (tasks completed)": [day["completed"]
                                                     for day in over_time],
                "avg latency (ms)": [day["avg_latency_ms"]
                                     for day in over_time],
            }, index=[day["date"] for day in over_time])
            st.line_chart(frame)
        else:
            st.info("Not enough data - no evaluation days recorded yet.")

    st.markdown("---")
    st.subheader("Pilot testers")
    testers = metrics.get("testers") or []
    if testers:
        st.dataframe(pd.DataFrame(testers), use_container_width=True,
                     hide_index=True)
    else:
        st.info("Not enough data - no tester-tagged questions recorded yet.")

    st.subheader("Suggested pilot questions")
    ok, catalog = call_api("GET", "/evaluation/suggested-questions")
    if not ok:
        st.error(str(catalog))
    else:
        for entry in catalog["questions"]:
            st.markdown(f"- **{entry['id']}** ({entry['route_hint']}) — "
                        f"`{entry['question']}`")
        st.markdown(f"[Download the pilot CSV report]({API_URL}/evaluation/export)")


# ── Documents ───────────────────────────────────────────────────────────
def page_documents() -> None:
    st.header("Documents")
    st.caption("Policy documents for the assistant. Only .md and .txt files "
               "up to 2 MB (2,000,000 bytes) are accepted; the API enforces "
               "this exact byte limit. A repeat upload of the same file name "
               "replaces that document safely and never touches other "
               "documents' chunks.")

    ok, response = call_api("GET", "/health")
    if ok:
        docs = response["components"]["documents"]
        col1, col2, col3 = st.columns(3)
        col1.metric("Collection status",
                    "ready" if docs.get("available") else "not ingested")
        col2.metric("Chunks", f"{docs.get('chunks', 0):,}")
        col3.metric("Documents", f"{docs.get('documents', 0):,}")
        if not docs.get("available"):
            st.info(f"Document store status: {docs.get('detail', '')}")
        embedding = docs.get("embedding") or {}
        if embedding:
            verdict = "ready" if embedding.get("available") else "degraded"
            st.caption(f"Embedding provider ({embedding.get('provider')}, "
                       f"{embedding.get('model')}): **{verdict}** — "
                       f"{embedding.get('detail', '')}")
        corpus = docs.get("corpus_files") or []
        if corpus:
            with st.expander(f"Documents on disk ({len(corpus)})"):
                st.write(corpus)
    else:
        st.error(str(response))

    st.markdown("---")
    upload = st.file_uploader("Upload a policy document (.md, .txt)",
                              type=("md", "txt"),
                              accept_multiple_files=False)
    if upload is not None and st.button("Upload and index document"):
        content = upload.getvalue()
        if len(content) > MAX_DOCUMENT_UPLOAD_BYTES:
            st.error("The selected file is larger than 2 MB "
                     "(2,000,000 bytes).")
        else:
            with st.spinner("Storing and indexing..."):
                ok, response = call_api(
                    "POST", "/documents/upload",
                    timeout=DEFAULT_TIMEOUT_SECONDS,
                    files={"file": (upload.name, content)})
            if ok:
                if response.get("ingested"):
                    st.success(
                        f"{response['document']} indexed: "
                        f"{response['chunks']} chunks in "
                        f"{response['collection']} (total now "
                        f"{response['total_in_collection']}).")
                else:
                    st.warning(
                        f"{response['document']} was stored but not indexed "
                        f"yet: {response.get('detail', '')}. It will be "
                        "picked up by the next full ingestion.")
            else:
                st.error(str(response))


# ── History ─────────────────────────────────────────────────────────────
def page_history() -> None:
    st.header("History")
    current = ensure_session()
    st.caption("Transcripts are held in the API process only - restarting "
               "the API clears them. Browsing another session does not "
               "change the conversation you are having.")

    ok, response = call_api("GET", "/history", params={
        "session_id": current, "include_sessions": True})
    if not ok:
        st.error(str(response))
        return

    sessions = response.get("sessions") or []
    current_turns = response.get("turns") or []
    choices = {f"{current} (current session)": current}
    for summary in sessions:
        if summary["session_id"] != current:
            choices[f"{summary['session_id']} — "
                    f"{summary.get('turns', 0)} turns"] = summary["session_id"]

    if len(choices) > 1:
        picked = st.selectbox("Browse a previous session", list(choices),
                              index=0)
        viewed = choices[picked]
    else:
        viewed = current

    if viewed == current:
        turns = current_turns
        badge = "current session"
    else:
        ok, other = call_api("GET", "/history", params={"session_id": viewed})
        if not ok:
            st.error(str(other))
            return
        turns = other.get("turns") or []
        badge = "previous session"
        if st.button("Make this the session for new chats"):
            st.session_state["session_id"] = viewed
            st.session_state["messages"] = []
            st.rerun()

    st.markdown(f"Session `{viewed}` · {badge} · {len(turns)} turns")
    if not turns:
        st.info("This session has no messages yet - ask the assistant "
                "something.")
        return
    for turn in reversed(turns):        # most recent first
        with st.expander(f"{turn.get('index', '?')} · "
                         f"[{turn.get('route', '?')}] "
                         f"{turn.get('question', '')}"):
            st.markdown(turn.get("answer", ""))
            citations = turn.get("citations") or []
            if citations:
                st.caption("Cited: " + ", ".join(str(item) for item in
                                                 citations))


# ── Sidebar ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Navigation")
    page = st.radio("Go to", PAGES, label_visibility="collapsed")

    st.markdown("---")
    st.subheader("Backend")
    st.caption(f"API: {API_URL} (override with the ERP_API_URL setting)")
    if st.button("Check API health"):
        ok, response = call_api("GET", "/health", timeout=10)
        if ok:
            st.success(f"API online — v{response['version']}")
        else:
            st.error(str(response))
    if st.session_state.get("session_id"):
        st.caption(f"Conversation: `{st.session_state['session_id'][:8]}…`")
    if st.button("Start a new conversation"):
        st.session_state["session_id"] = uuid.uuid4().hex
        st.session_state["messages"] = []
        st.rerun()

# ── Page dispatch ───────────────────────────────────────────────────────
if page == "Overview":
    page_overview()
elif page == "AI Assistant":
    page_assistant()
elif page == "Business Analytics":
    page_analytics()
elif page == "Evaluation & Performance":
    page_evaluation()
elif page == "Documents":
    page_documents()
elif page == "History":
    page_history()
