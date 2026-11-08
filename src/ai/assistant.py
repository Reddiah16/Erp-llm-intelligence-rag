"""Turning one question into one answer: route, gather, synthesise, verify.

This is the only module that touches the database, the document store and the
model for a chat question, and each of those is optional at runtime. The
matrix matters more than the happy path:

| SQL ran | documents found | model answered | what the response says |
|---------|-----------------|----------------|------------------------|
| yes     | -                 | yes            | figures from the results |
| no      | -                 | -              | "figure unavailable", no number |
| yes     | -                 | no             | the raw query result, rendered in code |
| -       | yes               | no             | the retrieved passages, quoted |
| -       | no                | -              | "policy documents unavailable" |

`data_available` and `documents_available` are reported separately from the
answer text because the brief requires no fabricated metrics when data is
unavailable, and a reader (or a test) can only check that claim if the response
states the inputs it had. `unverified_figures` is the same idea applied to the
output: every number in the answer is looked for in the evidence the answer was
built from, and one that is not there is named rather than quietly shipped.
"""
import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional, Tuple

from src.ai import routing, sql_tools
from src.ai.llm import LlmClient, LlmProtocolError, LlmUnavailable
from src.ai.settings import AiSettings, default_settings
from src.db import config as db_config
from src.llm.prompts import ANALYST_PROMPT
from src.rag import documents

# Shown where no structured data was queried, so the model cannot mistake an
# empty section for a zero.
NO_SQL_CONTEXT = ("No structured data was queried for this question, so there "
                  "are no business figures available here. Do not state a "
                  "metric; answer from the policy documents only.")

_DATE_TEXT = re.compile(r"\d{4}-\d{2}-\d{2}")
_NUMBER_TEXT = re.compile(r"\d[\d,]*(?:\.\d+)?")
CENTS = Decimal("0.01")
_LIST_QUERY_NAMES = {
    "top_products_by_revenue",
    "top_customers_by_revenue",
    "products_below_safety_stock",
    "products_needing_reorder",
}


@dataclass(frozen=True)
class Answer:
    """Everything one exchange produced, including what could not be produced."""

    question: str
    text: str
    route: str
    classifier: str
    reason: str = ""
    query: Optional[str] = None
    sql: Optional[Dict[str, Any]] = None
    citations: List[Dict[str, Any]] = field(default_factory=list)
    data_available: bool = False
    documents_available: bool = False
    generated: bool = False
    model: str = ""
    refusals: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    unverified_figures: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.text,
            "route": self.route,
            "classifier": self.classifier,
            "reason": self.reason,
            "query": self.query,
            "sql": self.sql,
            "citations": list(self.citations),
            "data_available": self.data_available,
            "documents_available": self.documents_available,
            "generated": self.generated,
            "model": self.model,
            "refusals": list(self.refusals),
            "warnings": list(self.warnings),
            "unverified_figures": list(self.unverified_figures),
        }


def _figures(text: str) -> set:
    """Business figures in a block, thousands separators removed.

    Small integers without a decimal part are skipped on purpose: "1." and
    "2." in a numbered list are not metrics, and treating them as fabricated
    figures would throw away good answers. A number with a decimal part is
    always considered, because a policy threshold can be 1.5 days.
    """
    stripped = _DATE_TEXT.sub(" ", text or "")
    found = set()
    for token in _NUMBER_TEXT.findall(stripped):
        cleaned = token.replace(",", "")
        integer, _, fraction = cleaned.partition(".")
        if not fraction and len(integer.lstrip("0")) < 3:
            continue
        if cleaned.replace(".", "").lstrip("0"):
            found.add(cleaned)
    return found


def _decimal(value: str) -> Optional[Decimal]:
    try:
        return Decimal(value)
    except InvalidOperation:
        return None


def verify_figures(answer: str, evidence: str) -> List[str]:
    """Numbers the answer states that the evidence does not contain.

    Two forms count as contained: the digits as written, and the same value
    rounded to cents, because a model that turns 4400592.800400 into
    4,400,592.80 has formatted the figure rather than invented it. Anything
    else - a rescaled amount, a count that is not the row count, a figure from
    another question - is returned for the caller to act on.
    """
    available = _figures(evidence)
    rounded = {value.quantize(CENTS) for value
               in (_decimal(item) for item in available)
               if value is not None}
    unverified = []
    for value in sorted(_figures(answer), key=lambda item: (-len(item), item)):
        if value in available:
            continue
        number = _decimal(value)
        if number is not None and any(number == candidate
                                       for candidate in rounded):
            continue
        unverified.append(value)
    return unverified


# ------------------------------------------------------------------ gathering

def open_connection(settings: AiSettings,
                    connection=None) -> Tuple[Any, Optional[str], bool]:
    """(connection, error, opened_here) - never raises.

    Public because the API needs the same rule for its own endpoints: a
    database that cannot be reached is an answer with a reason, not a stack
    trace. The DSN is reported in its redacted form.
    """
    if connection is not None:
        return connection, None, False
    try:
        configuration = db_config.config_from_env()
    except RuntimeError as exc:
        return None, str(exc), False
    try:
        return db_config.connect(configuration), None, True
    except Exception as exc:  # noqa: BLE001 - reported as unavailable
        return None, (f"could not connect to PostgreSQL "
                      f"({type(exc).__name__}) at "
                      f"{configuration.safe_dsn}"), False


def _run_sql(decision: routing.RouteDecision, settings: AiSettings,
             connection=None) -> Tuple[Optional[sql_tools.QueryResult], str]:
    """Execute the approved query the router named, if any."""
    if not decision.needs_sql or decision.query is None:
        return None, ""
    conn, error, opened_here = open_connection(settings, connection)
    if conn is None:
        return None, error
    try:
        result = sql_tools.execute(conn, decision.query, decision.params,
                                   limit_ceiling=settings.limit_ceiling)
        return result, ""
    except sql_tools.UnknownQuery as exc:
        # Unreachable through routing, which validates against the same map.
        return None, str(exc)
    except ValueError as exc:
        return None, f"the requested row count was refused: {exc}"
    except Exception as exc:  # noqa: BLE001 - reported as unavailable
        return None, (f"the approved query could not run "
                      f"({type(exc).__name__})")
    finally:
        if opened_here and conn is not None:
            try:
                conn.rollback()
                conn.close()
            except Exception:  # noqa: BLE001 - closing cannot fail an answer
                pass


def _run_documents(question: str, decision: routing.RouteDecision,
                   settings: AiSettings,
                   top_k: Optional[int]
                   ) -> Tuple[List[Dict[str, Any]], str]:
    """Retrieve policy passages, or explain why there are none."""
    if not decision.needs_documents:
        return [], ""
    try:
        hits = documents.search(question, settings, top_k)
    except Exception as exc:  # noqa: BLE001 - reported as unavailable
        return [], (f"the document store could not be queried "
                    f"({type(exc).__name__})")
    if hits:
        return hits, ""
    reported = documents.status(settings)
    return [], (reported["reason"] or
                "no passage in the policy corpus matched this question")


# ------------------------------------------------------------------- contexts

def _sql_context(result: Optional[sql_tools.QueryResult], error: str,
                 settings: AiSettings) -> str:
    if result is not None:
        return result.render(max_rows=settings.max_rows_in_context)
    return f"UNAVAILABLE: {error or 'no query was run'}. There is no figure to " \
           f"report, so state that the figure is unavailable."


def _grain(result: Optional[sql_tools.QueryResult]) -> str:
    return result.grain if result is not None else "none"


def _evidence(sql_context: str, document_context: str) -> str:
    """The exact text the model was handed, and therefore the only text a
    figure in its answer may come from.

    Earlier this rebuilt the rows from the result object, which lost the
    "69 rows returned" header the render added - and then flagged a correct
    count as fabricated. Verification must read what the model read.
    """
    return f"{sql_context}\n{document_context}"


def _append_complete_sql_rows(answer: str,
                              result: Optional[sql_tools.QueryResult]) -> str:
    """Keep list answers complete when the model omits returned SQL rows."""
    if result is None or result.name not in _LIST_QUERY_NAMES or not result.rows:
        return answer

    text = answer.casefold()
    labels = [
        str(row.get("name") or row.get("account_number")
            or row.get("product_number") or "")
        for row in result.rows
    ]
    if labels and all(label.casefold() in text for label in labels if label):
        return answer

    rows = [
        "- " + " | ".join(f"{column}={value}"
                           for column, value in row.items())
        for row in result.rows
    ]
    complete = (f"Complete approved SQL result rows "
                f"({result.row_count} returned):\n" + "\n".join(rows))
    return f"{answer.rstrip()}\n\n{complete}"


def _fallback_text(route: str, result: Optional[sql_tools.QueryResult],
                   error: str, citations: List[Dict[str, Any]], reason: str,
                   settings: AiSettings,
                   note: str = ("No language model was available, so this "
                                "answer is the retrieved evidence itself rather "
                                "than a summary of it.")) -> str:
    """An answer written by this module instead of a model.

    Everything here is copied from a query result or a retrieved passage, so the
    degraded path cannot invent a number that the primary path did not have.
    """
    if route == "unsupported":
        catalogue = "\n".join(f"  - {title}"
                              for title in sql_tools.catalogue_titles())
        return ("I could not match that to an approved query or to the policy "
                f"documents. {reason or 'No route applied.'}\n"
                f"Structured questions I can answer:\n{catalogue}\n"
                "Policy questions I can answer come from the documents under "
                "documents/ (inventory policy, procurement guidelines, "
                "business rules, data dictionary).")

    parts = [note]
    if result is not None:
        parts.append(result.render(max_rows=settings.max_answer_rows))
    elif route in ("sql", "hybrid"):
        parts.append(f"Business figures are unavailable: {error}")
    if citations:
        quoted = "\n".join(f"[{item['id']}] {item['document']} - "
                           f"{item['section']}\n{item['excerpt']}"
                           for item in citations)
        parts.append(f"Policy documents retrieved:\n{quoted}")
    elif route in ("rag", "hybrid"):
        parts.append(f"Policy documents are unavailable: {error or reason}")
    return "\n\n".join(parts)


# ------------------------------------------------------------------ answering

def answer_question(question: str, settings: Optional[AiSettings] = None,
                    llm: Optional[LlmClient] = None,
                    decision: Optional[routing.RouteDecision] = None,
                    connection: Any = None,
                    top_k: Optional[int] = None) -> Answer:
    """One question in, one auditable answer out.

    `decision` and `connection` exist for callers that already routed or already
    connected - the dashboard reuses one connection per page render, and tests
    inject a decision so routing failure cannot mask an orchestration failure.
    """
    settings = settings or default_settings()
    text = (question or "").strip()
    client = llm or LlmClient(settings)

    if decision is None:
        decision = routing.classify(text, settings, client)

    if not text:
        return Answer(question=text, text=_fallback_text(
            "unsupported", None, "", [], "the question was empty", settings),
            route="unsupported", classifier=decision.classifier,
            reason="the question was empty")

    result, sql_error = _run_sql(decision, settings, connection)
    hits, document_error = _run_documents(text, decision, settings, top_k)

    citations: List[Dict[str, Any]] = []
    if decision.needs_documents:
        document_context, citations = documents.render_context(hits)
        if not hits:
            document_context = (f"POLICY DOCUMENTS UNAVAILABLE: "
                                f"{document_error}. State that the policy "
                                f"could not be retrieved.")
    else:
        document_context = NO_SQL_CONTEXT

    sql_context = _sql_context(result, sql_error, settings)

    evidence = _evidence(sql_context, document_context)
    unverified: List[str] = []
    warnings: List[str] = []

    if decision.route == "unsupported":
        answer_text = _fallback_text("unsupported", result, sql_error,
                                     citations, decision.reason, settings)
        generated = False
    else:
        prompt = ANALYST_PROMPT.format(
            sql_grain=_grain(result),
            sql_context=sql_context,
            document_context=document_context,
            question=text)
        answer_text = ""
        generated = False
        try:
            answer_text = client.complete(prompt).strip()
        except (LlmUnavailable, LlmProtocolError) as exc:
            warnings.append(f"the answer was rendered without a model: {exc}")
        if answer_text:
            generated = True
        else:
            answer_text = _fallback_text(decision.route, result, sql_error,
                                         citations, decision.reason, settings)
            if not warnings or "rendered without a model" not in warnings[-1]:
                warnings.append("the model returned no text, so the retrieved "
                                "evidence is shown instead")

        answer_text = _append_complete_sql_rows(answer_text, result)
        unverified = verify_figures(answer_text, evidence)
        if unverified:
            # A figure the evidence does not contain is a fabrication, and an
            # apology in the warning field does not make the wrong number less
            # wrong to the person reading only the answer. The evidence is
            # rendered instead, and what was dropped is named.
            warnings.append(
                "the model's wording was discarded because it stated figures "
                f"that are not in the evidence: {', '.join(unverified[:8])}")
            answer_text = _fallback_text(
                decision.route, result, sql_error, citations, decision.reason,
                settings,
                note=("The model's wording stated a figure that the retrieved "
                      "evidence does not contain, so the evidence is shown "
                      "instead of an answer that cannot be trusted."))
            generated = False

    if result is None and decision.needs_sql and sql_error:
        warnings.append(f"structured data unavailable: {sql_error}")
    if decision.needs_documents and not hits and document_error:
        warnings.append(f"policy documents unavailable: {document_error}")

    return Answer(
        question=text,
        text=answer_text,
        route=decision.route,
        classifier=decision.classifier,
        reason=decision.reason,
        query=decision.query,
        sql=result.to_dict(settings.max_answer_rows) if result else None,
        citations=citations,
        data_available=result is not None,
        documents_available=bool(hits),
        generated=generated,
        model=settings.chat_model,
        refusals=list(decision.refusals),
        warnings=warnings,
        unverified_figures=unverified)
