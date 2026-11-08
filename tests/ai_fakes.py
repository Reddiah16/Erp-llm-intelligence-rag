"""Offline doubles for the Phase 3 AI layer.

Every Phase 3 test runs without Ollama, PostgreSQL, a downloaded
embedding model or the network. The replacements here stand in for the
three things the real stack reaches for:

* `FakeLlm`           - what `src/ai/llm.LlmClient` offers to routing and
  answering, scripted per test, recording what it was handed.
* `FakeDatabase`      - what `psycopg.connect` should offer to
  `src/db/analytics`: rows served from the test, `execute` recorded,
  `description` names taken from the dict keys, exactly the cursor shape
  analytics._rows reads.
* `FakeEmbeddingFunction` - deterministic bag-of-word vectors so
  ChromaDB retrieval asserts are stable with no ONNX model on disk.

Nothing here is mocked deeper than one boundary: routing validity,
figure verification, chunking, citations and the store are all real code
running against these doubles.
"""
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.ai.settings import AiSettings

# ------------------------------------------------------------------ settings


def make_settings(tmp_path: Optional[Path],
                  override_chat_model: Optional[str] = None) -> AiSettings:
    """Hermetic settings: temp ChromaDB and documents dir, one fixed model.

    `tmp_path=None` is accepted for tests that never touch the document
    store or the vector store (routing, orchestration, connections): the
    paths exist but are never written to.
    """
    chat = override_chat_model or "fake-model"
    base = Path(".") if tmp_path is None else tmp_path
    return AiSettings(
        ollama_base_url="http://localhost:11434",
        chat_model=chat,
        route_model=chat,
        embedding_provider="chroma",
        embedding_model="nomic-embed-text",
        chroma_path=base / "chroma",
        documents_dir=base / "documents",
        document_collection="test_collection",
        top_k=4,
        max_rows_in_context=15,
        max_answer_rows=15,
        limit_ceiling=50,
        llm_timeout=30,
        availability_ttl=60,
        max_output_tokens=400,
    )


# ------------------------------------------------------------------------ llm


class FakeLlm:
    """LlmClient's public surface, scripted: replies in order, probes fixed.

    `available=False` makes availability() report the server as down, which
    routes through the heuristic fallback, and complete() raises - the two
    degradation paths every Phase 3 caller must survive.
    """

    def __init__(self, replies: List[str], available: bool = True,
                 unavailability_reason: str = "Ollama at http://localhost:11434 "
                                              "is not reachable (ConnectionError)") -> None:
        self.replies = list(replies)
        self.available = available
        self.unavailability_reason = unavailability_reason
        self.prompts: List[str] = []
        self.generate_kwargs: List[Dict[str, Any]] = []

    # the interface routing.classify and assistant.answer_question use
    def availability(self, model: Optional[str] = None,
                     refresh: bool = False) -> Tuple[bool, str]:
        if self.available:
            return True, f"{model or 'fake-model'} is available"
        return False, self.unavailability_reason

    def complete(self, prompt: str, **kwargs: Any) -> str:
        self.prompts.append(prompt)
        self.generate_kwargs.append(kwargs)
        if not self.available:
            from src.ai.llm import LlmUnavailable
            raise LlmUnavailable("unavailable by script")
        if not self.replies:
            return ""
        return self.replies.pop(0)

    # the interface src/ai/llm.LlmClient users rely on through the real class
    def list(self) -> Dict[str, Any]:
        return {"models": [{"name": "fake-model"}]}

    def generate(self, **kwargs: Any) -> Dict[str, Any]:
        self.generate_kwargs.append(kwargs)
        text = self.replies.pop(0) if self.replies else ""
        return {"response": text}


# ------------------------------------------------------------------ database


class FakeResult:
    """The one piece of a psycopg cursor analytics._rows reads."""

    def __init__(self, rows: List[Dict[str, Any]]) -> None:
        self.rows = rows
        self.description = [type("Column", (), {"name": key})()
                            for key in (rows[0] if rows else {})]

    def fetchall(self) -> List[Tuple]:
        return [tuple(row.values()) for row in self.rows]

    def fetchone(self) -> Optional[Tuple]:
        return self.fetchall()[0] if self.rows else None


class FakeDatabase:
    """A connection that answers SELECTs from canned rows.

    Rows are matched by a substring of the SQL (`pattern`), so one test
    can serve the revenue query and the pending-order count from the same
    open "connection". Nothing is parsed: the test name is the only
    guarantee of which canned rows a query will see.
    """

    def __init__(self, responses: List[Tuple[str, List[Dict[str, Any]]]]) -> None:
        self.responses = responses
        self.executed: List[Tuple[str, Any]] = []
        self.closed = False
        self.rolled_back = False

    def execute(self, sql: str, params: Any = None) -> FakeResult:
        self.executed.append((sql, params))
        for pattern, rows in self.responses:
            if pattern.lower() in sql.lower():
                return FakeResult(rows)
        raise AssertionError(
            f"FakeDatabase has no canned rows for SQL beginning {sql[:80]!r}")

    def rollback(self) -> None:
        self.rolled_back = True

    def close(self) -> None:
        self.closed = True


# --------------------------------------------------------------- embeddings

VOCABULARY = (
    "safety", "stock", "reorder", "order", "pending", "revenue",
    "product", "policy", "purchase", "document", "definition", "rule",
    "return", "calendar", "days", "refund", "business", "approval",
    "shipping", "charge", "refundable", "damaged",
)

VECTOR_SIZE = len(VOCABULARY)


def fake_vector(text: str) -> List[float]:
    """A deterministic bag-of-words vector over a fixed vocabulary.

    Cosine similarity on this is a overlap measure: strings that share
    vocabulary rank first, irrelevant strings rank low. Fixed and
    dependency-free, so retrieval assertions never flake on a model
    update or a missing cache.
    """
    lowered = (text or "").lower()
    counts = [1.0 if word in lowered else 0.0 for word in VOCABULARY]
    if not any(counts):
        # Nothing in vocabulary: still deterministic, orthogonal-ish to all
        # vocabulary vectors so irrelevant passages rank last.
        digest = int(hashlib.sha256(lowered.encode("utf-8")).hexdigest(), 16)
        counts[digest % VECTOR_SIZE] = 1.0
    return counts


class FakeEmbeddingFunction:
    """Chroma's embedding-function contract with fixed vectors."""

    def __call__(self, input: List[str]) -> List[List[float]]:  # noqa: A002
        return [fake_vector(text) for text in input]

    def embed_documents(self, input: List[str]) -> List[List[float]]:
        return self(input)

    def embed_query(self, input: str) -> List[float]:
        return self([input])[0]


def install_fake_embeddings(monkeypatch) -> None:
    """Point every ChromaDB collection at FakeEmbeddingFunction.

    Two seams are needed. The provider factory is patched for when
    documents.get_collection attaches a function on creation. And because
    chroma binds `ef.DefaultEmbeddingFunction()` as the Collection
    constructor default at import time, a collection opened without one
    (search opens with create=False) would fall back to the real ONNX
    model, so the constructor is wrapped to inject the fake there too.
    No ONNX model is ever reached for.
    """
    from chromadb.utils import embedding_functions
    from chromadb.api.models import Collection as collection_module

    monkeypatch.setattr(embedding_functions, "DefaultEmbeddingFunction",
                        lambda: FakeEmbeddingFunction())
    real_init = collection_module.Collection.__init__

    def always_fake(self, *args, **kwargs):
        kwargs["embedding_function"] = FakeEmbeddingFunction()
        return real_init(self, *args, **kwargs)

    monkeypatch.setattr(collection_module.Collection, "__init__", always_fake)
