"""Pilot evaluation tracking: what the assistant actually did, from the app only.

Every /chat answer records one turn (route, duration, evidence flags, the anonymous
tester id) and every evaluator/user grades it via /feedback. Both are appended as
JSON events under a dedicated directory - the ERP database tables, schema and role
are untouched, and evaluation storage survives API restarts because it is on disk.

Compliance: AI_ACT_COMPLIANCE.md promises query inputs are not stored beyond the
current session. This module keeps that promise - the raw question text is never
persisted; a turn carries a stable `question_id` derived from a hash of the
session and question, or a catalog id (PQ-*) when the question is one of the
curated pilot questions, whose texts live in SUGGESTED_QUESTIONS.

The metrics follow the brief exactly and never invent figures: a percentage is
None ("Not enough data") whenever its denominator is zero.
"""
import csv
import hashlib
import io
import json
import os
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parents[2]

EVALUATION_DATA_DIR_ENV = "EVALUATION_DATA_DIR"
DEFAULT_DATA_DIR = "evaluation_data"

TURNS_FILE = "turns.jsonl"
FEEDBACK_FILE = "feedback.jsonl"
EXPORT_FILE = "pilot_report.csv"

TESTER_IDS = tuple(f"Tester {index:02d}" for index in range(1, 11))

CORRECT_VALUES = ("yes", "no", "not_sure")
CITATION_VALUES = ("yes", "no", "not_applicable")

# The short set every pilot tester works through. The ids are the stable
# question identifiers that end up in records, so evaluation reports can point
# at "PQ-SQL-01" without storing the question text.
SUGGESTED_QUESTIONS: Tuple[Dict[str, str], ...] = (
    {"id": "PQ-SQL-01", "question": "How many purchase orders are pending?",
     "route_hint": "SQL"},
    {"id": "PQ-SQL-02", "question": "Which products are below safety stock level?",
     "route_hint": "SQL"},
    {"id": "PQ-SQL-03", "question": "What is the total revenue from sales orders?",
     "route_hint": "SQL"},
    {"id": "PQ-RAG-01", "question": "What is the rule for products below safety stock?",
     "route_hint": "RAG"},
    {"id": "PQ-RAG-02", "question": "What is the receiving tolerance rule?",
     "route_hint": "RAG"},
    {"id": "PQ-HYB-01", "question": "How many purchase orders are pending, and what "
                                    "is the rule for approving them?",
     "route_hint": "Hybrid"},
)

_CATALOG = {entry["question"]: entry["id"] for entry in SUGGESTED_QUESTIONS}

CSV_FIELDS = ("timestamp", "tester_id", "session_id", "question_id",
              "question_kind", "route", "classifier", "duration_ms",
              "sql_executed", "sql_success", "citations_count",
              "data_available", "documents_available", "generated",
              "unverified_figures", "warnings", "api_cost", "turn_id",
              "helpful", "correct", "citations_relevant", "feedback_comment")


def utc_now() -> str:
    """One timestamp format everywhere: ISO-8601 UTC, millisecond precision."""
    now = datetime.now(timezone.utc)
    return now.isoformat(timespec="milliseconds")


def new_turn_id() -> str:
    """A fresh id per answered question event."""
    return uuid.uuid4().hex[:24]


def question_id_for(session_id: str, question: str) -> Tuple[str, str]:
    """A stable, non-reversible question identifier plus its kind.

    Curated pilot questions map to their catalog id, so a report can say
    "PQ-RAG-01"; anything else gets a hash of the session id and the exact
    question text - deterministic for repeats, unreadable without the
    original inputs, and never a stored question.
    """
    catalog_id = _CATALOG.get((question or "").strip())
    if catalog_id:
        return catalog_id, "catalog"
    digest = hashlib.sha256()
    digest.update((session_id or "").encode("utf-8"))
    digest.update(b"\x00")
    digest.update((question or "").strip().encode("utf-8"))
    return digest.hexdigest()[:16], "hash"


def is_local_inference(base_url: str) -> bool:
    """Whether the configured LLM is a local server - API cost is then confidently zero."""
    try:
        from urllib.parse import urlparse
        hostname = urlparse(base_url or "").hostname or ""
    except ValueError:
        return False
    return hostname.lower() in ("localhost", "127.0.0.1", "host.docker.internal",
                                "0.0.0.0", "::1")


@dataclass(frozen=True)
class TurnRecord:
    """One answered question, as the evaluation store persists it."""

    turn_id: str
    session_id: str
    question_id: str
    question_kind: str
    tester_id: str = ""
    route: str = ""
    classifier: str = ""
    asked_at: str = ""
    duration_ms: Optional[int] = None
    sql_executed: bool = False
    sql_success: bool = False
    citations_count: int = 0
    data_available: bool = False
    documents_available: bool = False
    generated: bool = False
    unverified_figures: int = 0
    warnings: int = 0
    api_cost: Optional[float] = None
    cost_note: str = ""

    def as_record(self) -> Dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "session_id": self.session_id,
            "question_id": self.question_id,
            "question_kind": self.question_kind,
            "tester_id": self.tester_id,
            "route": self.route,
            "classifier": self.classifier,
            "asked_at": self.asked_at,
            "duration_ms": self.duration_ms,
            "sql_executed": self.sql_executed,
            "sql_success": self.sql_success,
            "citations_count": self.citations_count,
            "data_available": self.data_available,
            "documents_available": self.documents_available,
            "generated": self.generated,
            "unverified_figures": self.unverified_figures,
            "warnings": self.warnings,
            "api_cost": self.api_cost,
            "cost_note": self.cost_note,
        }


@dataclass(frozen=True)
class FeedbackRecord:
    """The evaluator's grading of one turn (self-contained, event-folded)."""

    turn_id: str
    session_id: str = ""
    question_id: str = ""
    tester_id: str = ""
    helpful: Optional[bool] = None
    correct: Optional[str] = None
    citations_relevant: Optional[str] = None
    comment: Optional[str] = None
    at: str = ""

    def as_record(self) -> Dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "session_id": self.session_id,
            "question_id": self.question_id,
            "tester_id": self.tester_id,
            "helpful": self.helpful,
            "correct": self.correct,
            "citations_relevant": self.citations_relevant,
            "comment": self.comment,
            "at": self.at,
        }


def validate_feedback_fields(helpful, correct, citations_relevant, comment) -> None:
    """Reject wrong shapes at the edge, before anything is appended."""
    if helpful is not None and not isinstance(helpful, bool):
        raise ValueError("helpful must be true, false or omitted")
    if correct is not None and correct not in CORRECT_VALUES:
        raise ValueError(f"correct must be one of {CORRECT_VALUES}, "
                         f"never a bare 'unsure' or a number")
    if citations_relevant is not None and citations_relevant not in CITATION_VALUES:
        raise ValueError(f"citations_relevant must be one of {CITATION_VALUES}")
    if comment is not None:
        if not isinstance(comment, str):
            raise ValueError("comment must be a string")
        if len(comment) > 1000:
            raise ValueError("the comment is longer than 1000 characters")
    if helpful is None and correct is None and citations_relevant is None \
            and (comment is None or not comment.strip()):
        raise ValueError("the feedback is empty - rate at least one criterion "
                         "or write a comment")


class PilotStore:
    """Append-only JSON event store for turns and feedback, plus its metrics."""

    def __init__(self, directory: Optional[Path] = None) -> None:
        raw = os.getenv(EVALUATION_DATA_DIR_ENV) or DEFAULT_DATA_DIR
        path = Path(raw)
        self.directory = path if path.is_absolute() else ROOT_DIR / raw
        if directory is not None:                 # tests inject deterministic dirs
            self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    # ------------------------------------------------------------ persistence

    def _append(self, filename: str, record: Dict[str, Any]) -> None:
        with self._lock:
            with (self.directory / filename).open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def record_turn(self, turn: TurnRecord) -> str:
        self._append(TURNS_FILE, turn.as_record())
        return turn.turn_id

    def record_feedback(self, feedback: FeedbackRecord) -> Dict[str, Any]:
        self._append(FEEDBACK_FILE, feedback.as_record())
        return self.feedback_for_turn(feedback.turn_id) or feedback.as_record()

    def _read(self, filename: str) -> List[Dict[str, Any]]:
        path = self.directory / filename
        if not path.exists():
            return []
        import json
        records: List[Dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    records.append(json.loads(line))
                except ValueError:
                    pass  # a torn last line is skipped, older event lines stand
        return records

    def turns(self) -> List[Dict[str, Any]]:
        return self._read(TURNS_FILE)

    def feedback_by_turn(self) -> Dict[str, Dict[str, Any]]:
        """Latest feedback event per turn, folded in append order."""
        folded: Dict[str, Dict[str, Any]] = {}
        for record in self._read(FEEDBACK_FILE):
            identifier = record.get("turn_id")
            if identifier:
                folded[identifier] = record
        return folded

    def feedback_for_turn(self, turn_id: str) -> Optional[Dict[str, Any]]:
        return self.feedback_by_turn().get(turn_id)

    def turn(self, turn_id: str) -> Optional[Dict[str, Any]]:
        """One turn by id. Feedback is folded on top by the caller when needed."""
        for turn in self.turns():
            if turn.get("turn_id") == turn_id:
                return turn
        return None

    def wipe(self) -> None:                    # honoured by tests only
        with self._lock:
            for filename in (TURNS_FILE, FEEDBACK_FILE):
                (self.directory / filename).unlink(missing_ok=True)

    # ---------------------------------------------------------------- metrics

    @staticmethod
    def _percent(numerator: int, sample: int) -> Optional[float]:
        return None if sample <= 0 else round(100.0 * numerator / sample, 1)

    def summary_metrics(self) -> Dict[str, Any]:
        """The seven metrics plus breakdowns, with 'not enough data' as None."""
        turns = [turn for turn in self.turns() if turn.get("turn_id")]
        feedback = self.feedback_by_turn()

        evaluated, correct_yes, correct_no, citations_yes, citations_no = \
            0, 0, 0, 0, 0
        helpful_yes = helpful_no = correct_na = citations_na = 0
        graded: Dict[str, Dict[str, List[int]]] = {}
        for turn in turns:
            grade = feedback.get(turn["turn_id"])
            if not grade:
                continue
            evaluated += 1
            route = turn.get("route", "")
            bucket = graded.setdefault(route, {"evaluated": 0, "yes": 0})
            if grade.get("helpful") is True:
                helpful_yes += 1
            elif grade.get("helpful") is False:
                helpful_no += 1
            if grade.get("correct") == "yes":
                correct_yes += 1
                bucket["evaluated"] += 1; bucket["yes"] += 1
            elif grade.get("correct") == "no":
                correct_no += 1
                bucket["evaluated"] += 1
            elif grade.get("correct") == "not_sure":
                correct_na += 1
            if grade.get("citations_relevant") == "yes":
                citations_yes += 1
            elif grade.get("citations_relevant") == "no":
                citations_no += 1
            elif grade.get("citations_relevant") == "not_applicable":
                citations_na += 1

        sql_executed = [turn for turn in turns if turn.get("sql_executed")]
        sql_success = [turn for turn in sql_executed if turn.get("sql_success")]
        timed = [turn for turn in turns
                 if isinstance(turn.get("duration_ms"), int)]
        latency = [turn["duration_ms"] for turn in timed]

        supported = [
            turn for turn in turns
            if turn.get("generated")
            and turn.get("route") in ("sql", "rag", "hybrid")
            and (turn.get("sql_success") or turn.get("citations_count", 0) > 0)
        ]
        answerable = [turn for turn in turns if turn.get("generated")
                      and turn.get("route") in ("sql", "rag", "hybrid")]

        costs = [turn["api_cost"] for turn in turns
                 if isinstance(turn.get("api_cost"), (int, float))]
        unknown_cost = sum(1 for turn in turns if turn.get("api_cost") is None)
        if not turns:
            cost_note = ""
        elif costs:
            cost_note = "API cost only; infrastructure cost is not included"
        else:
            cost_note = "cost not instrumented for this provider - no invented figure"

        by_route: Dict[str, Dict[str, Any]] = {}
        for route in ("sql", "rag", "hybrid", "unsupported"):
            stats = graded.get(route, {})
            by_route[route] = {
                "value": self._percent(stats.get("yes", 0),
                                       stats.get("evaluated", 0)),
                "sample": stats.get("evaluated", 0),
            }

        by_date: Dict[str, Dict[str, List[int]]] = {}
        latency_series = []
        for turn in sorted(turns, key=lambda item: (item.get("asked_at", ""),
                                                   item.get("turn_id", ""))):
            date = (turn.get("asked_at") or "")[:10]
            bucket = by_date.setdefault(date, {"evaluated": [],
                                               "completed": [], "latency": []})
            bucket["evaluated"].append(1)
            grade = feedback.get(turn["turn_id"])
            bucket["completed"].append(1 if grade and grade.get("helpful") else 0)
            if isinstance(turn.get("duration_ms"), int):
                bucket["latency"].append(turn["duration_ms"])
            if isinstance(turn.get("duration_ms"), int):
                latency_series.append(
                    {"ts": turn.get("asked_at", ""),
                     "ms": turn["duration_ms"]})

        over_time = []
        for date in sorted(by_date):
            bucket = by_date[date]
            turns_day = len(bucket["evaluated"])
            completed = sum(bucket["completed"])
            over_time.append({
                "date": date,
                "evaluated": turns_day,
                "completed": completed,
                "avg_latency_ms": round(sum(bucket["latency"]) /
                                        len(bucket["latency"]))
                                  if bucket["latency"] else None,
            })

        accuracy_series = []
        running_evaluated = 0
        running_correct = 0
        for turn in sorted(turns, key=lambda item: (item.get("asked_at", ""),
                                                   item.get("turn_id", ""))):
            grade = feedback.get(turn["turn_id"])
            if not grade or grade.get("correct") not in ("yes", "no"):
                continue
            running_evaluated += 1
            if grade.get("correct") == "yes":
                running_correct += 1
            accuracy_series.append({
                "ts": turn.get("asked_at", ""),
                "value": round(100.0 * running_correct / running_evaluated, 1),
            })
        latency_series = latency_series[-500:]

        tester_summary: Dict[str, Dict[str, List[int]]] = {}
        for turn in turns:
            tester = turn.get("tester_id") or ""
            bucket = tester_summary.setdefault(
                tester, {"questions": [], "helpful_yes": [], "correct": []})
            bucket["questions"].append(1)
            grade = feedback.get(turn["turn_id"])
            if grade:
                if grade.get("helpful") is True:
                    bucket["helpful_yes"].append(1)
                elif grade.get("helpful") is False:
                    bucket["helpful_yes"].append(0)
                if grade.get("correct") in ("yes", "no"):
                    bucket["correct"].append(grade.get("correct") == "yes")
        testers = [{
            "tester_id": tester if tester else "(not set)",
            "questions": len(stats["questions"]),
            "helpful": sum(stats["helpful_yes"]),
            "helpful_judged": len(stats["helpful_yes"]),
            "accuracy": self._percent(sum(1 for ok in stats["correct"] if ok),
                                      len(stats["correct"])),
            "correct_judged": len(stats["correct"]),
        } for tester, stats in sorted(tester_summary.items())]

        return {
            "generated_at": utc_now(),
            "total_questions": len(turns),
            "evaluated_turns": evaluated,
            "answer_accuracy": {"value": self._percent(correct_yes,
                                                       correct_yes + correct_no),
                                "sample": correct_yes + correct_no},
            "sql_execution_success": {"value": self._percent(len(sql_success),
                                                             len(sql_executed)),
                                      "sample": len(sql_executed)},
            "average_latency_ms": {"value": round(sum(latency) / len(latency), 1)
                                             if latency else None,
                                   "sample": len(latency)},
            "retrieval_accuracy": {"value": self._percent(citations_yes,
                                                          citations_yes + citations_no),
                                   "sample": citations_yes + citations_no},
            "data_grounding": {"value": self._percent(len(supported),
                                                      len(answerable)),
                               "sample": len(answerable)},
            "cost_per_query": {"value": round(sum(costs) / len(costs), 2)
                                         if costs else None,
                               "sample": len(costs),
                               "unknown_count": unknown_cost,
                               "note": cost_note},
            "task_completion": {"value": self._percent(helpful_yes,
                                                       helpful_yes + helpful_no),
                                "sample": helpful_yes + helpful_no},
            "accuracy_by_route": by_route,
            "feedback": {"helpful_yes": helpful_yes, "helpful_no": helpful_no,
                         "correct_yes": correct_yes, "correct_no": correct_no,
                         "correct_not_sure": correct_na,
                         "citations_yes": citations_yes,
                         "citations_no": citations_no,
                         "citations_not_applicable": citations_na,
                         "comments": sum(1 for grade in feedback.values()
                                         if (grade.get("comment") or "").strip()),
                         "turns": evaluated},
            "over_time": over_time,
            "latency_series": latency_series,
            "accuracy_series": accuracy_series[-500:],
            "testers": testers,
        }

    def export_csv(self, path: Optional[Path] = None) -> Path:
        """The pilot report: one row per turn, feedback folded in."""
        target = Path(path) if path is not None else \
            self.directory / EXPORT_FILE
        with target.open("w", encoding="utf-8", newline="") as handle:
            self._write_csv(csv.DictWriter(handle, fieldnames=CSV_FIELDS))
        return target

    def export_csv_text(self) -> str:
        """Render the report without changing any persisted evaluation file."""
        handle = io.StringIO(newline="")
        self._write_csv(csv.DictWriter(handle, fieldnames=CSV_FIELDS))
        return handle.getvalue()

    def _write_csv(self, writer: csv.DictWriter) -> None:
        """Write the current turns and their latest feedback to a CSV writer."""
        feedback = self.feedback_by_turn()
        rows = sorted(self.turns(),
                      key=lambda item: (item.get("asked_at", ""),
                                        item.get("turn_id", "")))
        writer.writeheader()
        for turn in rows:
            grade = feedback.get(turn.get("turn_id"), {})
            writer.writerow({
                "timestamp": turn.get("asked_at", ""),
                "tester_id": turn.get("tester_id", ""),
                "session_id": turn.get("session_id", ""),
                "question_id": turn.get("question_id", ""),
                "question_kind": turn.get("question_kind", ""),
                "route": turn.get("route", ""),
                "classifier": turn.get("classifier", ""),
                "duration_ms": turn.get("duration_ms"),
                "sql_executed": turn.get("sql_executed"),
                "sql_success": turn.get("sql_success"),
                "citations_count": turn.get("citations_count"),
                "data_available": turn.get("data_available"),
                "documents_available": turn.get("documents_available"),
                "generated": turn.get("generated"),
                "unverified_figures": turn.get("unverified_figures"),
                "warnings": turn.get("warnings"),
                "api_cost": turn.get("api_cost"),
                "turn_id": turn.get("turn_id", ""),
                "helpful": grade.get("helpful"),
                "correct": grade.get("correct"),
                "citations_relevant": grade.get("citations_relevant"),
                "feedback_comment": grade.get("comment"),
            })


_CACHED: Optional[PilotStore] = None


def default_store() -> PilotStore:
    """The process store, built once so every call lands in one directory."""
    global _CACHED
    if _CACHED is None:
        _CACHED = PilotStore()
    return _CACHED


def reset_store() -> None:
    """Forget the cached instance (tests and CLI use this between directories)."""
    global _CACHED
    _CACHED = None
