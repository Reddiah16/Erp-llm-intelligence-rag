"""Session-scoped conversation history, held in process memory only.

AI_ACT_COMPLIANCE.md states that "query inputs are not logged or stored beyond
the current session", so this module keeps exchanges in a bounded dictionary and
writes nothing to disk. A SQLite table would be more convenient across restarts
and would contradict the compliance statement in the same repository, so the
limitation is deliberate: restarting the API clears the transcript.

Both bounds are real. An unbounded per-session list leaks memory for a long-lived
process, and an unbounded session dictionary turns a demo endpoint into a way to
exhaust RAM, so the oldest session is evicted when the cap is reached.
"""
import threading
import uuid
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Deque, Dict, List, Optional

DEFAULT_MAX_SESSIONS = 64
DEFAULT_MAX_TURNS = 20

SESSION_ID_LENGTH = 32


@dataclass(frozen=True)
class Exchange:
    """One completed question and answer, as the transcript stores it."""

    index: int
    question: str
    answer: str
    route: str
    classifier: str
    query: Optional[str]
    citations: List[str] = field(default_factory=list)
    asked_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "question": self.question,
            "answer": self.answer,
            "route": self.route,
            "classifier": self.classifier,
            "query": self.query,
            "citations": list(self.citations),
            "asked_at": self.asked_at,
        }


def new_session_id() -> str:
    """An opaque identifier for a browser session or API client."""
    return uuid.uuid4().hex[:SESSION_ID_LENGTH]


def normalise_session_id(value: Optional[str]) -> str:
    """Accept a caller's id or mint one, without trusting what was sent.

    A caller-supplied identifier is used only as a dictionary key, but an
    unbounded or whitespace-filled key from a request body is a memory lever, so
    the length and character set are enforced rather than assumed.
    """
    cleaned = (value or "").strip()
    if not cleaned:
        return new_session_id()
    if len(cleaned) > SESSION_ID_LENGTH:
        raise ValueError(f"session_id must be {SESSION_ID_LENGTH} characters "
                         f"or fewer")
    if not all(character.isalnum() or character in "-_" for character in cleaned):
        raise ValueError("session_id may contain only letters, digits, dash "
                         "and underscore")
    return cleaned


class HistoryStore:
    """Bounded transcripts per session, safe for a threaded server."""

    def __init__(self, max_sessions: int = DEFAULT_MAX_SESSIONS,
                 max_turns: int = DEFAULT_MAX_TURNS) -> None:
        if max_sessions <= 0 or max_turns <= 0:
            raise ValueError("history bounds must be positive")
        self.max_sessions = max_sessions
        self.max_turns = max_turns
        self._sessions: "OrderedDict[str, Deque[Exchange]]" = OrderedDict()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ writes

    def add(self, session_id: str, question: str, answer: str, route: str,
            classifier: str = "", query: Optional[str] = None,
            citations: Optional[List[Any]] = None) -> Exchange:
        """Append one exchange and return what was stored.

        `citations` may be the citation dicts from src/ai/assistant.py or plain
        reference strings; only the reference is kept, because the stored
        transcript should not grow to hold document text it does not display.
        """
        references = [_reference(item) for item in (citations or [])]
        with self._lock:
            turns = self._sessions.get(session_id)
            if turns is None:
                turns = deque(maxlen=self.max_turns)
                self._sessions[session_id] = turns
                while len(self._sessions) > self.max_sessions:
                    self._sessions.popitem(last=False)
            exchange = Exchange(
                index=len(turns),
                question=(question or "").strip(),
                answer=(answer or "").strip(),
                route=route,
                classifier=classifier,
                query=query,
                citations=[reference for reference in references if reference],
                asked_at=datetime.now().isoformat(timespec="seconds"))
            turns.append(exchange)
            self._sessions.move_to_end(session_id)
            return exchange

    def clear(self, session_id: str) -> int:
        """Drop one session's transcript, returning how many turns went."""
        with self._lock:
            turns = self._sessions.pop(session_id, None)
            return len(turns) if turns else 0

    # ------------------------------------------------------------------- reads

    def turns(self, session_id: str) -> List[Exchange]:
        with self._lock:
            return list(self._sessions.get(session_id, ()))

    def transcript(self, session_id: str) -> List[Dict[str, Any]]:
        return [exchange.to_dict() for exchange in self.turns(session_id)]

    def last(self, session_id: str) -> Optional[Exchange]:
        turns = self.turns(session_id)
        return turns[-1] if turns else None

    def sessions(self) -> List[Dict[str, Any]]:
        """Session summaries, most recently used first, without question text.

        The list endpoint is an operator's view of activity. Replaying stored
        questions to an unauthenticated caller would be a disclosure, so this
        reports counts and timestamps only.
        """
        with self._lock:
            items = list(self._sessions.items())
        summaries = []
        for session_id, turns in reversed(items):
            last = turns[-1] if turns else None
            summaries.append({
                "session_id": session_id,
                "turns": len(turns),
                "last_route": last.route if last else "",
                "last_asked_at": last.asked_at if last else "",
            })
        return summaries

    def __len__(self) -> int:
        with self._lock:
            return len(self._sessions)

    def __contains__(self, session_id: str) -> bool:
        with self._lock:
            return session_id in self._sessions


def _reference(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("id") or item.get("reference") or "")
    return str(item or "")


_STORE: Optional[HistoryStore] = None


def default_store() -> HistoryStore:
    """The process-wide transcript store.

    Cached like src/ai/settings.py caches its settings: one server process
    should answer for one history, and rebuilding it per request would make
    GET /history return an empty list moments after a question was asked.
    """
    global _STORE
    if _STORE is None:
        _STORE = HistoryStore()
    return _STORE


def reset_store() -> None:
    """Forget every transcript. Used by tests, and by nothing else."""
    global _STORE
    _STORE = None
