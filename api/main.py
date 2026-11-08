import os
import time
from typing import Any, Dict, List, Literal, Optional

from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from src.rag.chain import query_rag
from src.evaluation.metrics import log_query_metrics

load_dotenv()

# Phase 3 imports sit below the Phase 1 ones rather than beside them. chromadb
# and mlflow both pull in ONNX and numpy binaries, and the working pin order in
# requirements.dev.txt was established with the Phase 1 modules loading first;
# reordering imports here would be an unforced risk of the import-time crash
# that pin exists to prevent.
from src.ai import assistant, history, llm, routing, sql_tools
from src.ai.settings import AiSettings, default_settings
from src.db import config as db_config
from src.rag import documents
from src.evaluation import pilot

# Kept at 1.1.0 rather than 2.0.0 on purpose: /query still behaves as it did in
# Phase 1 and the Phase 3 endpoints are additions, so a client written against
# the earlier contract does not need rewriting.
app = FastAPI(
    title       = "ERP LLM Intelligence API",
    description = "Natural language business intelligence over the ERP "
                  "PostgreSQL schema and the policy document corpus",
    version     = "1.1.0"
)

# The HTTP bound on the /analytics row count. src.ai.sql_tools.clamp_limit
# applies the configured AI_LIMIT_CEILING afterwards, so a request under this
# number can still be clamped, and a request over it is refused at the edge.
MAX_LIMIT_PARAMETER = 1000


def settings() -> AiSettings:
    """The process configuration, as a dependency so tests can override it."""
    return default_settings()


def database(settings: AiSettings = Depends(settings)):
    """A read-only connection per request, or a 503 that says why.

    The transaction is never committed and the connection is always closed:
    every query this API can run is a SELECT from src/ai/sql_tools.py, and a
    FastAPI dependency is the one place that can guarantee the cleanup.
    """
    connection, error, opened_here = assistant.open_connection(settings)
    if connection is None:
        raise HTTPException(status_code=503, detail=error)
    try:
        yield connection
    finally:
        if opened_here:
            connection.rollback()
            connection.close()


def safe_detail(exc: Exception) -> str:
    """An error message that may be returned to a client.

    Exception text from a database driver can carry the connection string it
    tried, and a connection string carries a password. The type and the first
    line are enough to debug from, so the message is truncated and reported
    through the redacted helpers wherever this module builds it itself.
    """
    text = " ".join(str(exc).split())[:240]
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__

# Table keyword mapping for metric logging — keys map onto the canonical
# AdventureWorks table names produced by src/chunking/chunker.py
TABLE_KEYWORDS = {
    "stock":      "stock_levels",
    "safety":     "stock_levels",
    "reorder":    "stock_levels",
    "inventory":  "stock_levels",
    "material":   "product_master",
    "product":    "product_master",
    "purchase":   "purchase_orders",
    "vendor":     "purchase_orders",
    "customer":   "customer_master",
    "revenue":    "sales_orders",
    "order":      "sales_orders",
    "territory":  "sales_orders",
}


def infer_relevant_table(question: str) -> str:
    q = question.lower()
    for keyword, table in TABLE_KEYWORDS.items():
        if keyword in q:
            return table
    return "sales_orders"


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    question: str
    answer:   str
    sources:  list


class ChatRequest(BaseModel):
    """One natural language question.

    The length bound is deliberately wider than the router's own limit: a
    question over the router limit still gets an "unsupported" answer that
    explains the rule, while an absurd payload is refused here with a readable
    validation error instead of being sent to a model.

    `tester_id` is the pilot's anonymous identity ("Tester 01".."Tester 10"),
    stored with the evaluation record and never resolved to a person.
    """
    question:   str = Field(min_length=1, max_length=5000)
    session_id: Optional[str] = Field(default=None, max_length=32)
    tester_id:  Optional[str] = Field(default=None, max_length=32)


class ChatResponse(BaseModel):
    question:            str
    answer:              str
    route:               str
    classifier:          str
    reason:              str
    query:               Optional[str]
    sql:                 Optional[Dict[str, Any]]
    citations:           List[Dict[str, Any]]
    data_available:      bool
    documents_available: bool
    generated:           bool
    model:               str
    refusals:            List[str]
    warnings:            List[str]
    unverified_figures:  List[str]
    session_id:          str
    # evaluation instrumentation (see src/evaluation/pilot.py)
    turn_id:            Optional[str] = None
    question_id:        Optional[str] = None
    duration_ms:        Optional[int] = None
    api_cost_inr:       Optional[float] = None
    cost_note:          str = ""
    tester_id:          str = ""


class FeedbackRequest(BaseModel):
    """The evaluator's grading of one answered turn.

    Every field is optional; at least one must be present. `turn_id` selects
    the exact exchange and comes from the /chat response, so feedback can
    never attach to the wrong question.
    """
    turn_id:            str = Field(min_length=1, max_length=64)
    helpful:            Optional[bool] = None
    correct:            Optional[Literal["yes", "no", "not_sure"]] = None
    citations_relevant: Optional[Literal["yes", "no", "not_applicable"]] = None
    comment:            Optional[str] = Field(default=None, max_length=1000)
    tester_id:          Optional[str] = Field(default=None, max_length=32)


class FeedbackResponse(BaseModel):
    recorded:     bool
    turn_id:      str
    question_id:  str
    feedback:     Dict[str, Any]


class AnalyticsResponse(BaseModel):
    limit:    int
    queries:  List[Dict[str, Any]]
    unavailable: List[Dict[str, str]] = []


class UploadResponse(BaseModel):
    document:          str
    chunks:            int
    collection:        str
    total_in_collection: int
    ingested:          bool
    detail:            str = ""


class HistoryResponse(BaseModel):
    session_id: str
    turns:      List[Dict[str, Any]]
    sessions:   List[Dict[str, Any]]


@app.get("/health")
def health(current: AiSettings = Depends(settings)):
    """Liveness plus what the assistant can actually reach right now.

    Every component check is wrapped, because /health that returns 500 when
    PostgreSQL is down tells a caller less than /health that says the service
    is up and the database is unreachable.
    """
    components: Dict[str, Any] = {}
    try:
        reachable, reason = db_config.server_reachable()
        components["database"] = {"available": reachable, "detail": reason}
    except Exception as exc:  # noqa: BLE001 - reported, never fatal here
        components["database"] = {"available": False,
                                  "detail": safe_detail(exc)}
    try:
        usable, reason = llm.LlmClient(current).availability(current.route_model)
        components["llm"] = {"available": usable, "detail": reason,
                             "model": current.route_model}
    except Exception as exc:  # noqa: BLE001
        components["llm"] = {"available": False, "detail": safe_detail(exc)}
    try:
        reported = documents.status(current)
        components["documents"] = {
            "available": reported["available"],
            "chunks": reported["chunks"],
            "documents": reported["documents"],
            "detail": reported["reason"] or "ready",
            "corpus_files": reported["corpus_files"],
            # Provider readiness, checked without loading or downloading an
            # embedding model. `available` above still describes the actual
            # store; this says what the embedding side of it can do now.
            "embedding": reported.get("embedding") or {
                "provider": current.embedding_provider,
                "model": current.embedding_model,
                "available": False,
                "detail": "unknown - run /health again",
            },
        }
    except Exception as exc:  # noqa: BLE001
        components["documents"] = {"available": False,
                                   "detail": safe_detail(exc)}
    components["queries_allowed"] = len(sql_tools.ALLOWLIST)
    return {"status": "ok", "version": app.version, "components": components}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    try:
        result = query_rag(req.question)

        # Log retrieval metrics to MLflow
        log_query_metrics(
            question       = req.question,
            sources        = result["sources"],
            relevant_table = infer_relevant_table(req.question),
            answer_length  = len(result["answer"])
        )

        return QueryResponse(
            question = result["question"],
            answer   = result["answer"],
            sources  = result["sources"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=safe_detail(e))


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, current: AiSettings = Depends(settings)):
    """The whole assistant: classify, gather from SQL and documents, answer.

    A database or model that is down is not an HTTP error here. The response
    still carries an answer, with data_available and documents_available saying
    which inputs were missing, because "the figure is unavailable" is the
    correct answer to the question and a 503 is not.
    """
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    started = time.perf_counter()
    try:
        session_id = history.normalise_session_id(req.session_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    try:
        answer = assistant.answer_question(question, current)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 - the caller gets a readable fault
        raise HTTPException(status_code=500, detail=safe_detail(exc))

    duration_ms = int((time.perf_counter() - started) * 1000)
    # Evaluation instrumentation runs beside the exchange and must never break
    # it: any storage failure yields a null turn_id rather than a failed API.
    turn = None
    try:
        question_id, question_kind = pilot.question_id_for(session_id, question)
        local_inference = pilot.is_local_inference(current.ollama_base_url)
        turn = pilot.TurnRecord(
            turn_id=pilot.new_turn_id(),
            session_id=session_id,
            question_id=question_id,
            question_kind=question_kind,
            tester_id=(req.tester_id or "").strip(),
            route=answer.route,
            classifier=answer.classifier,
            asked_at=pilot.utc_now(),
            duration_ms=duration_ms,
            sql_executed=answer.route in ("sql", "hybrid"),
            sql_success=bool(answer.sql) and answer.data_available,
            citations_count=len(answer.citations or []),
            data_available=answer.data_available,
            documents_available=answer.documents_available,
            generated=answer.generated,
            unverified_figures=len(answer.unverified_figures or []),
            warnings=len(answer.warnings or []),
            api_cost=0.0 if local_inference else None,
            cost_note=("local Ollama inference - API cost only, "
                       "infrastructure not included"
                       if local_inference else
                       "the provider's API cost is not instrumented; "
                       "no invented figure is stored"),
        )
        pilot.default_store().record_turn(turn)
    except Exception:  # noqa: BLE001 - evaluation hints off, answering stands
        turn = None

    body = answer.to_dict()
    body["session_id"] = session_id
    body["turn_id"] = turn.turn_id if turn else None
    body["question_id"] = turn.question_id if turn else None
    body["duration_ms"] = duration_ms
    body["api_cost_inr"] = (turn.api_cost if turn else None)
    body["cost_note"] = (turn.cost_note if turn else
                         "the query was not evaluated (turn recording failed)")
    body["tester_id"] = turn.tester_id if turn else (req.tester_id or "")
    history.default_store().add(
        session_id, question=answer.question, answer=answer.text,
        route=answer.route, classifier=answer.classifier,
        query=answer.query, citations=answer.citations)
    return ChatResponse(**body)


@app.post("/documents/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...),
                          current: AiSettings = Depends(settings)):
    """Store one policy document and add it to the retrieval corpus.

    The name and the bytes are validated before anything is written, and a file
    that cannot be read as UTF-8 text is refused rather than ingested as noise.
    """
    content = await file.read()
    if len(content) > 2_000_000:
        raise HTTPException(status_code=413,
                            detail="the uploaded document is larger than 2 MB")
    try:
        path = documents.add_uploaded_document(content, file.filename or "",
                                               current)
    except ValueError as exc:
        # A rejected name or encoding is the client's mistake, not the server's.
        raise HTTPException(status_code=400, detail=str(exc))

    try:
        summary = documents.ingest(current, paths=[path])
    except Exception as exc:  # noqa: BLE001 - the file is stored regardless
        # Reporting the storage without the embedding keeps the failure honest:
        # the document is on disk and will be picked up by the next ingestion,
        # but it is not searchable yet.
        return UploadResponse(document=path.name, chunks=0,
                              collection=current.document_collection,
                              total_in_collection=0, ingested=False,
                              detail=safe_detail(exc))
    return UploadResponse(document=path.name, chunks=summary["chunks"],
                          collection=summary["collection"],
                          total_in_collection=summary["total_in_collection"],
                          ingested=True, detail="ready")


@app.get("/analytics", response_model=AnalyticsResponse)
def analytics_endpoint(
        question: Optional[str] = Query(
            default=None, description="one approved query name"),
        limit: int = Query(default=10, ge=1, le=MAX_LIMIT_PARAMETER),
        current: AiSettings = Depends(settings),
        connection=Depends(database)):
    """The approved queries directly, with no model in the loop.

    This is the endpoint a dashboard or a script should use for figures: the
    same allowlist the router is held to, the same grain metadata, and no text
    generation that could restate a number incorrectly.

    `limit` is bounded twice on purpose: 422 for a nonsense value at the edge of
    the API, then src.ai.sql_tools.clamp_limit against the configured ceiling
    before the number reaches a query, so no caller can ask for a million rows
    by quoting a larger parameter.
    """
    names = ([question] if question else sql_tools.tool_names())
    if question and question not in sql_tools.ALLOWLIST:
        raise HTTPException(
            status_code=404,
            detail=f"{question!r} is not an approved query; choose from "
                   f"{sql_tools.tool_names()}")

    results, unavailable = [], []
    for name in names:
        try:
            result = sql_tools.execute(connection, name, limit=limit,
                                       limit_ceiling=current.limit_ceiling)
            results.append(result.to_dict(current.max_answer_rows))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except Exception as exc:  # noqa: BLE001 - reported per query
            unavailable.append({"query": name, "detail": safe_detail(exc)})
    return AnalyticsResponse(limit=limit, queries=results,
                             unavailable=unavailable)


@app.get("/history", response_model=HistoryResponse)
def history_endpoint(
        session_id: Optional[str] = Query(default=None, max_length=32),
        include_sessions: bool = Query(default=False)):
    """The transcript for one session, held in this process only.

    AI_ACT_COMPLIANCE.md promises that query inputs are not stored beyond the
    current session, so there is no database table behind this and no history
    survives a restart. Session summaries are opt-in and carry counts only.
    """
    try:
        identifier = history.normalise_session_id(session_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    store = history.default_store()
    return HistoryResponse(
        session_id=identifier,
        turns=store.transcript(identifier),
        sessions=store.sessions() if include_sessions else [])


# ---------------------------------------------------------------- evaluation
# The pilot endpoints read from the same store the /chat handler writes into
# (src/evaluation/pilot.py: JSON event files, never the ERP tables).

@app.post("/feedback", response_model=FeedbackResponse)
def feedback_endpoint(req: FeedbackRequest, current: AiSettings = Depends(settings)):
    """Grade one recorded exchange.

    The turn_id comes from the /chat response, so feedback attaches to the
    exact question it grades. A repeat submission replaces the earlier grades
    field by field - the caller sends only what changed and earlier values
    stand. An empty payload (no rating, no comment) is refused.
    """
    try:
        pilot.validate_feedback_fields(req.helpful, req.correct,
                                       req.citations_relevant, req.comment)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    store = pilot.default_store()
    turn = store.turn(req.turn_id)
    if turn is None:
        raise HTTPException(
            status_code=404,
            detail=f"no recorded turn {req.turn_id!r} - /feedback grades the "
                   f"turn_id returned by a /chat answer")
    existing = store.feedback_for_turn(req.turn_id) or {}
    record = pilot.FeedbackRecord(
        turn_id=req.turn_id,
        session_id=turn.get("session_id", ""),
        question_id=turn.get("question_id", ""),
        tester_id=(req.tester_id or turn.get("tester_id")
                   or existing.get("tester_id") or ""),
        helpful=(req.helpful if req.helpful is not None
                 else existing.get("helpful")),
        correct=(req.correct if req.correct is not None
                 else existing.get("correct")),
        citations_relevant=(req.citations_relevant
                            if req.citations_relevant is not None
                            else existing.get("citations_relevant")),
        comment=(req.comment if req.comment is not None
                 else existing.get("comment")),
        at=pilot.utc_now(),
    )
    merged = store.record_feedback(record)
    return FeedbackResponse(recorded=True, turn_id=req.turn_id,
                            question_id=turn.get("question_id", ""),
                            feedback=merged)


@app.get("/evaluation/metrics")
def evaluation_metrics():
    """The seven pilot metrics with their sample sizes, never a fabricated value."""
    return pilot.default_store().summary_metrics()


@app.get("/evaluation/export")
def evaluation_export():
    """The pilot report as CSV: one row per recorded turn, feedback folded in."""
    content = pilot.default_store().export_csv_text()
    return Response(
        content=content,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=pilot_report.csv"},
    )


@app.get("/evaluation/suggested-questions")
def evaluation_suggested():
    """The curated pilot question set, and the anonymous tester identities."""
    return {"questions": list(pilot.SUGGESTED_QUESTIONS),
            "testers": list(pilot.TESTER_IDS),
            "note": "the tester ids are anonymous labels - they are never "
                    "resolved to a person"}