"""Deciding between SQL, documents and both - and refusing what fits neither.

The model classifies; this module decides what is allowed to happen. That
ordering is the security property the brief asks for: a model output is an
untrusted suggestion that must name a query from src.ai.sql_tools.ALLOWLIST
before anything runs. There is no path from a model's text to the database.

Two things are reported on every decision rather than hidden:

* `classifier` - "llm", "heuristic" or "guard". When Ollama is stopped the
  assistant still routes with deterministic keyword rules, and the response
  says so, because a guess presented as a model judgement is the one failure
  mode a reviewer cannot see coming.
* `refusals` - each rule the payload broke and what was done about it, so a
  downgrade to "unsupported" is explained instead of merely felt.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import re

from src.ai import sql_tools
from src.ai.llm import LlmClient, LlmProtocolError, LlmUnavailable, looks_like_sql
from src.ai.settings import AiSettings, default_settings
from src.llm.prompts import ROUTE_PROMPT

ROUTES = ("sql", "rag", "hybrid", "unsupported")

# A classification is one small label block, so it gets a budget far below the
# answer budget: a router that spends 400 tokens deciding would be the slowest
# part of the request.
ROUTE_TOKENS = 120

# The router's output contract, field by field rather than as JSON. This was
# changed after measurement: with a JSON contract the local 1.5B model either
# copied the example object verbatim ("route": "sql|rag|hybrid|unsupported") or
# copied an example query name onto unrelated questions. Labelling each value on
# its own line removed both failure modes, and the parser below is the only
# place that turns those labels into a payload.
_LABEL_LINE = re.compile(
    r"^\s*(ROUTE|QUERY|LIMIT|REASON)\s*:\s*(.*)$", re.MULTILINE | re.IGNORECASE)
_EMPTY_VALUES = ("", "none", "null", "n/a", "nil")

# Questions longer than this are not classified: an oversized payload is either
# a paste of a report or an attempt to fill the prompt with it, and neither is
# answered usefully by routing.
GUARDED_MESSAGES = {
    "empty": "the question was empty",
    "too_long": "the question was longer than the {limit}-character limit",
}


@dataclass(frozen=True)
class RouteDecision:
    route: str
    query: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    classifier: str = "guard"
    refusals: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.route not in ROUTES:
            raise ValueError(f"not a route: {self.route!r}")

    @property
    def needs_sql(self) -> bool:
        return self.route in ("sql", "hybrid") and self.query is not None

    @property
    def needs_documents(self) -> bool:
        return self.route in ("rag", "hybrid")

    @property
    def executable(self) -> bool:
        """Whether this decision may reach the database."""
        return self.needs_sql and self.query in sql_tools.ALLOWLIST

    def to_dict(self) -> Dict[str, Any]:
        return {"route": self.route, "query": self.query,
                "params": dict(self.params), "reason": self.reason,
                "classifier": self.classifier, "refusals": list(self.refusals)}


def _route_of(payload: Dict[str, Any]) -> Optional[str]:
    """The route as a known value, or None when the model named something else."""
    raw = str(payload.get("route", "")).strip().lower()
    return raw if raw in ROUTES else None


def _named_query(text: str) -> Optional[str]:
    """One approved query name mentioned in free text, or None.

    Accepted only when exactly one allowlist name appears. This is not the model
    being allowed to choose: the candidate is already on the allowlist, the text
    is not executed, and a payload that names two queries is refused rather than
    guessed between. It exists because a small local model demonstrably
    sometimes writes "route": null while spelling the right query in its reason,
    and an assistant that answers "unsupported" in that situation is blaming the
    user for the model's formatting error.
    """
    found = [name for name in sql_tools.ALLOWLIST if name in text]
    return found[0] if len(found) == 1 else None


def _clean(value: str) -> str:
    return (value or "").strip().strip('"').strip("'").strip("`*). ")


def parse_classification(text: str) -> Dict[str, Any]:
    """The label block as a payload dict, ready for validate().

    Unknown field values are passed through rather than repaired: validate()
    owns every decision about what may run, and a parser that guesses would put
    a second, quieter policy in front of the allowlist check.
    """
    labels: Dict[str, str] = {}
    for key, value in _LABEL_LINE.findall(text or ""):
        labels.setdefault(key.lower(), _clean(value))
    if "route" not in labels:
        raise LlmProtocolError(
            f"the classifier returned no ROUTE line; reply began "
            f"{(text or '')[:120]!r}")

    payload: Dict[str, Any] = {"route": labels["route"].lower(),
                               "reason": labels.get("reason", "")}
    query = labels.get("query", "")
    payload["query"] = None if query.lower() in _EMPTY_VALUES else query

    limit = labels.get("limit", "")
    if limit.lower() not in _EMPTY_VALUES:
        payload["params"] = {"limit": limit}
    return payload


def validate(payload: Dict[str, Any]) -> RouteDecision:
    """Turn one model payload into a decision, rejecting whatever is not allowed.

    Called with anything the model produced: a valid payload, a payload naming
    a query that is not on the allowlist, or one containing SQL.
    """
    refusals: List[str] = []
    route = _route_of(payload)

    raw_query = payload.get("query")
    query = raw_query.strip() if isinstance(raw_query, str) else None

    if route is None:
        if query in sql_tools.ALLOWLIST:
            # A model that got the route label wrong but named an approved query
            # has said which source it wanted. Reading the label down to "sql"
            # keeps the allowlist in force and adds no capability: the only
            # thing the label could have granted beyond that is document
            # retrieval, which a hybrid route would have had anyway.
            refusals.append(f"unrecognised route {payload.get('route')!r}; read "
                            f"as 'sql' because the payload names the approved "
                            f"query {query!r}")
            route = "sql"
        else:
            return RouteDecision(route="unsupported", reason="the model did not "
                                 "name a recognised route", classifier="llm",
                                 refusals=[f"unrecognised route "
                                           f"{payload.get('route')!r}"])

    if route in ("sql", "hybrid"):
        if query is None or query == "null":
            recovered = _named_query(str(payload.get("reason") or ""))
            if recovered:
                refusals.append(f"{route} route named no query, so "
                                f"{recovered!r} was recovered from the "
                                f"model's reason text")
                query = recovered
            elif route == "hybrid":
                refusals.append("hybrid route had no approved query, so it "
                                "became a document question")
                route = "rag"
                query = None
            else:
                refusals.append("no query named")
                route = "unsupported"
                query = None
        elif query not in sql_tools.ALLOWLIST:
            detail = ("the model proposed SQL, which is never executed"
                      if looks_like_sql(query)
                      else f"{query!r} is not an approved query")
            refusals.append(detail)
            if route == "hybrid" and not looks_like_sql(query):
                route = "rag"
            else:
                route = "unsupported"
            query = None

    if route in ("rag", "unsupported"):
        # A query cannot survive into a route that must not touch the database.
        if query is not None:
            refusals.append("query ignored for a route that reads no database")
        query = None

    params: Dict[str, Any] = {}
    if query is not None:
        tool = sql_tools.ALLOWLIST[query]
        supplied = payload.get("params")
        if not isinstance(supplied, dict):
            if supplied not in (None, {}):
                refusals.append("params was not an object and was dropped")
        else:
            for key, value in supplied.items():
                if key not in tool.params:
                    refusals.append(f"{query} does not accept parameter {key!r}")
                    continue
                params[key] = value
    elif payload.get("params"):
        refusals.append("params ignored because no query runs")

    reason = payload.get("reason")
    if not isinstance(reason, str):
        reason = ""
    return RouteDecision(route=route, query=query, params=params,
                         reason=reason.strip()[:300], classifier="llm",
                         refusals=refusals)


# ------------------------------------------------------------------ heuristic
# Keyword rules, in this order, used only when no model is available. They are
# intentionally blunt: the point is a working, explainable fallback rather than
# a second classifier to maintain.

POLICY_HINTS = (
    "policy", "policies", "rule", "rules", "guideline", "procedure",
    "definition", "defined", "meaning", "what is the", "what are the",
    "what does", "how is", "how are", "how do", "how should", "when is",
    "when should", "why", "should", "must", "required", "requirement",
    "allowed", "permit", "approval", "threshold", "escalat", "tolerance",
    "govern", "accordance", "document", "difference between", "return",
    "refund", "refundable", "exchange", "damaged", "defective",
    "shipping charge", "business days", "calendar days",
)

_POLICY_TOPIC_HINTS = (
    "policy", "policies", "rule", "rules", "guideline", "procedure",
    "definition", "approval", "threshold", "escalat", "tolerance",
    "permit", "requirement", "return", "refund", "refundable", "exchange",
    "damaged", "defective", "shipping charge", "business days",
    "calendar days",
)

# (marker, query) pairs checked against the lowercased question, specific first:
# "pending purchase order" must be tested before "purchase order" or the count
# question would answer as the value question.
SQL_HINTS = (
    ("below safety", "products_below_safety_stock"),
    ("safety stock", "products_below_safety_stock"),
    ("reorder", "products_needing_reorder"),
    ("pending purchase order", "pending_purchase_order_count"),
    ("open purchase order", "pending_purchase_order_count"),
    ("purchase order", "pending_purchase_order_value"),
    ("top product", "top_products_by_revenue"),
    ("best selling", "top_products_by_revenue"),
    ("most revenue by product", "top_products_by_revenue"),
    ("product revenue", "top_products_by_revenue"),
    ("top customer", "top_customers_by_revenue"),
    ("best customer", "top_customers_by_revenue"),
    ("customer revenue", "top_customers_by_revenue"),
    ("by month", "monthly_sales_trends"),
    ("per month", "monthly_sales_trends"),
    ("monthly", "monthly_sales_trends"),
    ("trend", "monthly_sales_trends"),
    ("total revenue", "total_sales_revenue"),
    ("total sales", "total_sales_revenue"),
    ("revenue", "total_sales_revenue"),
)

# A question needs one of these before a query is even considered. Bare
# "which", "order" and "below" are deliberately absent: "what is the reorder
# point?" contains "order" and asks for a definition, and running an approved
# query for it would answer a different question with a confident number.
FIGURE_HINTS = ("how many", "how much", "total", "sum", "count", "top",
                "best", "list", "average", "trend", "revenue", "sales",
                "value of", "worth", "pending", "products below safety",
                "products needing reorder", "products need reordering",
                "which products", "who has", "show me", "identify")


def heuristic(question: str) -> RouteDecision:
    """Deterministic routing for when no model answers.

    A figure marker selects the query, a policy marker selects the documents,
    and both together select the hybrid route. With neither marker the answer is
    "unsupported": guessing an approved query for a question this function
    cannot place would produce a confident number that answers something else.
    """
    text = f" {question.lower()} "
    policy = any(hint in text for hint in POLICY_HINTS)
    figure = any(hint in text for hint in FIGURE_HINTS)

    query, matched = None, ""
    if figure:
        for marker, candidate in SQL_HINTS:
            if marker in text:
                query, matched = candidate, marker
                break
        if query is None:
            if policy:
                return RouteDecision(
                    route="rag",
                    reason="the figure request concerns policy details and has "
                           "no applicable approved query",
                    classifier="heuristic")
            # A figure question this table cannot place must not get an
            # unrelated approved query or an invented numeric answer.
            return RouteDecision(
                route="unsupported",
                reason="keyword rules found a figure request but no approved "
                       "query it matches", classifier="heuristic")

    if query and policy:
        route, basis = "hybrid", f"figure {matched!r} and a policy marker"
    elif query:
        route, basis = "sql", f"figure marker {matched!r}"
    elif policy:
        route, basis = "rag", "a policy marker"
    else:
        route, basis = "unsupported", "no known signal"

    return RouteDecision(
        route=route, query=query,
        reason=f"keyword rules matched {basis}", classifier="heuristic")


def _explicit_hybrid_intent(question: str) -> Optional[RouteDecision]:
    """Recover clear two-part requests when the model drops one evidence source."""
    text = f" {question.lower()} "
    conjunction = any(marker in text for marker in
                     (" and ", " also ", " plus ", " along with ",
                      " as well as "))
    policy = any(marker in text for marker in
                 (" policy", " policies", " rule", " rules", " guideline",
                  " procedure", " approval", " threshold", " tolerance"))
    figure_request = any(marker in text for marker in
                         (" identify ", " list ", " show ", " which products",
                          " how many ", " how much ", " top ", " count ",
                          " revenue "))
    if not conjunction or not policy or not figure_request:
        return None
    decision = heuristic(question)
    return decision if decision.route == "hybrid" else None


def _explicit_uploaded_document_intent(question: str) -> bool:
    """Recognize a corpus filename or an explicit uploaded-document reference."""
    text = question.lower()
    refers_to_filename = bool(
        re.search(r"(?<![\w])[\w.-]+\.(?:md|txt)\b", text))
    refers_to_upload = bool(re.search(r"\bupload(?:ed)?\b", text))
    refers_to_document = bool(re.search(r"\b(document|file|policy)\b", text))
    return refers_to_filename or (refers_to_upload and refers_to_document)


def classify(question: str, settings: Optional[AiSettings] = None,
             llm: Optional[LlmClient] = None) -> RouteDecision:
    """One guard pass, one model call, one validation. Never one execution."""
    settings = settings or default_settings()
    text = (question or "").strip()

    if not text:
        return RouteDecision(route="unsupported", reason=GUARDED_MESSAGES["empty"])
    if len(text) > settings.max_question_characters:
        return RouteDecision(
            route="unsupported",
            reason=GUARDED_MESSAGES["too_long"].format(
                limit=settings.max_question_characters))

    client = llm or LlmClient(settings)
    ok, reason = client.availability(settings.route_model)
    if not ok:
        decision = heuristic(text)
        decision.refusals.append(f"classifier unavailable: {reason}")
        return decision

    prompt = ROUTE_PROMPT.format(catalog=sql_tools.catalog(), question=text)
    try:
        raw = client.complete(prompt, model=settings.route_model,
                              max_tokens=ROUTE_TOKENS)
        payload = parse_classification(raw)
    except (LlmUnavailable, LlmProtocolError) as exc:
        decision = heuristic(text)
        decision.refusals.append(f"classifier failed: {exc}")
        return decision

    decision = validate(payload)
    hybrid = _explicit_hybrid_intent(text)
    if hybrid is not None and decision.route != "hybrid":
        return RouteDecision(
            route="hybrid",
            query=hybrid.query,
            params=(decision.params if decision.query == hybrid.query
                    else hybrid.params),
            reason="explicit policy and approved-query requests require both "
                   "evidence sources",
            classifier="heuristic",
            refusals=decision.refusals + [
                f"model selected {decision.route}; explicit two-part intent "
                "was routed as hybrid"],
        )
    if _explicit_uploaded_document_intent(text):
        return RouteDecision(
            route="rag",
            reason="the question explicitly refers to a corpus document",
            classifier="heuristic",
            refusals=decision.refusals + [
                f"model selected {decision.route}; explicit document "
                "reference was routed to retrieval"],
        )

    fallback = heuristic(text)
    if any(hint in text.lower() for hint in _POLICY_TOPIC_HINTS) \
            and fallback.route == "rag" and fallback.query is None:
        if decision.route != "rag":
            return RouteDecision(
                route="rag",
                reason="the question concerns policy information without an "
                       "applicable approved analytics query",
                classifier="heuristic",
                refusals=decision.refusals + [
                    f"model selected {decision.route}; policy intent was "
                    "routed to document retrieval"],
            )
    if (decision.route in ("sql", "hybrid")
            and fallback.route == "unsupported"):
        return RouteDecision(
            route="unsupported",
            reason="no approved analytics query matches the question",
            classifier="heuristic",
            refusals=decision.refusals + [
                f"model selected {decision.route} with {decision.query!r}, "
                "but deterministic intent checks found no supported query"],
        )
    return decision
