"""Configuration for the AI layer, read from the environment only.

Mirrors src/db/config.py in shape because the two problems are the same: a
setting that is wrong should say so at start-up rather than produce a strange
answer forty messages later. There are no secrets here - Ollama needs no key and
ChromaDB is a local directory - so nothing in this module has to be redacted.

Paths are anchored to the repository root, matching src/data/loader.py and
src/chunking/chunker.py, so the assistant behaves identically from any terminal
directory.
"""
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parents[2]

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"

# Latency decides this default, not preference. Measured on the development
# machine with the same routing prompt the classifier sends (JSON only,
# temperature 0): qwen2.5:1.5b answered in 1.7s warm, qwen3:8b in 36.7s and
# qwen3:4b in 76.9s, all three correctly. A chat interface that spends most of
# a minute classifying one question is not usable, so the fast model is the
# default and OLLAMA_MODEL remains an override.
DEFAULT_CHAT_MODEL = "qwen2.5:1.5b"

# The user chose ChromaDB's built-in embedding model for the document store:
# all-MiniLM-L6-V2 through onnxruntime, which is pinned at 1.16.3 in the
# requirements files precisely because chromadb's default embedding function
# loads it at import time. "ollama" is supported for parity with the Phase 1
# row-record pipeline.
DEFAULT_EMBEDDING_PROVIDER = "chroma"
EMBEDDING_PROVIDERS = ("chroma", "ollama")

DOCUMENT_COLLECTION = "erp_documents"
DEFAULT_TOP_K = 4

# Rows handed to the model, and rows shown in an answer, were both 40/25 until
# a hybrid question spent over two minutes generating text and then timed out.
# A local model's cost is roughly proportional to the context it reads, and a
# 69-row stock list does not need 40 rows in the prompt to be answered - the row
# count carries the figure, the first rows carry the names. The full result set
# still reaches the dashboard and the API, which read the rows, not the prompt.
DEFAULT_MAX_ROWS_IN_CONTEXT = 15
DEFAULT_MAX_ANSWER_ROWS = 15
DEFAULT_LIMIT_CEILING = 50

# Generation, not connectivity: 120s was measured to be short for a hybrid
# question with four retrieved passages attached.
DEFAULT_LLM_TIMEOUT = 180

# An unbounded generation is the difference between a six second answer and a
# timeout: measured on this machine, the same routing prompt returned in 1.7s
# while a hybrid analyst prompt with a 15-row table attached ran past 180s and
# was cut off mid-sentence. num_predict is the blunt fix, and the prompts ask
# for three to six sentences, so 400 tokens is generous rather than binding.
DEFAULT_MAX_OUTPUT_TOKENS = 400
DEFAULT_AVAILABILITY_TTL = 60
MAX_QUESTION_CHARACTERS = 500


def _integer(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        raise RuntimeError(f"{name} must be a whole number, got {raw!r}")


def _positive(name: str, default: int) -> int:
    value = _integer(name, default)
    if value <= 0:
        raise RuntimeError(f"{name} must be greater than zero, got {value}")
    return value


@dataclass(frozen=True)
class AiSettings:
    """Everything the AI layer needs to know about this machine."""

    ollama_base_url: str
    chat_model: str
    route_model: str
    embedding_provider: str
    embedding_model: str
    chroma_path: Path
    documents_dir: Path
    document_collection: str
    top_k: int
    max_rows_in_context: int
    max_answer_rows: int
    limit_ceiling: int
    llm_timeout: int
    availability_ttl: int
    max_output_tokens: int
    max_question_characters: int = MAX_QUESTION_CHARACTERS

    def __post_init__(self) -> None:
        """Refuse a nonsensical setting at construction, not at first query."""
        if self.embedding_provider not in EMBEDDING_PROVIDERS:
            raise RuntimeError(
                f"RAG_EMBEDDING_PROVIDER must be one of "
                f"{list(EMBEDDING_PROVIDERS)}, got {self.embedding_provider!r}")
        for name in ("top_k", "max_rows_in_context", "max_answer_rows",
                     "limit_ceiling", "llm_timeout", "availability_ttl",
                     "max_output_tokens", "max_question_characters"):
            value = getattr(self, name)
            if not isinstance(value, int) or value <= 0:
                raise RuntimeError(f"{name} must be a positive integer, "
                                   f"got {value!r}")
        if self.max_answer_rows > self.max_rows_in_context:
            raise RuntimeError(
                "max_answer_rows cannot exceed max_rows_in_context: the answer "
                "would display rows the model was never given, and a reader "
                "would assume they were both derived from the same context.")
        if not self.ollama_base_url.startswith(("http://", "https://")):
            raise RuntimeError(
                f"OLLAMA_BASE_URL must be an http(s) URL, "
                f"got {self.ollama_base_url!r}")

    def anchor(self, path: Path) -> Path:
        """Resolve a relative setting against the repository root."""
        return path if path.is_absolute() else ROOT_DIR / path


def _path(name: str, default: str) -> Path:
    return Path(os.getenv(name) or default)


def settings_from_env(override: Optional[str] = None) -> AiSettings:
    """Build the settings from the environment.

    `override` exists for tests and for the dashboard's status panel; it names
    the chat model only and changes nothing else.
    """
    chat_model = override or os.getenv("OLLAMA_MODEL") or DEFAULT_CHAT_MODEL
    return AiSettings(
        ollama_base_url=os.getenv("OLLAMA_BASE_URL") or DEFAULT_OLLAMA_BASE_URL,
        chat_model=chat_model,
        route_model=os.getenv("OLLAMA_ROUTE_MODEL") or chat_model,
        embedding_provider=(os.getenv("RAG_EMBEDDING_PROVIDER")
                            or DEFAULT_EMBEDDING_PROVIDER),
        embedding_model=os.getenv("OLLAMA_EMBED_MODEL") or "nomic-embed-text",
        chroma_path=_path("CHROMA_PATH", "chroma_db"),
        documents_dir=_path("DOCUMENTS_DIR", "documents"),
        document_collection=os.getenv("DOCUMENT_COLLECTION") or DOCUMENT_COLLECTION,
        top_k=_positive("RAG_TOP_K", DEFAULT_TOP_K),
        max_rows_in_context=_positive("AI_MAX_ROWS_IN_CONTEXT",
                                      DEFAULT_MAX_ROWS_IN_CONTEXT),
        max_answer_rows=_positive("AI_MAX_ANSWER_ROWS", DEFAULT_MAX_ANSWER_ROWS),
        limit_ceiling=_positive("AI_LIMIT_CEILING", DEFAULT_LIMIT_CEILING),
        llm_timeout=_positive("OLLAMA_REQUEST_TIMEOUT", DEFAULT_LLM_TIMEOUT),
        availability_ttl=_positive("AI_AVAILABILITY_TTL", DEFAULT_AVAILABILITY_TTL),
        max_output_tokens=_positive("AI_MAX_OUTPUT_TOKENS",
                                    DEFAULT_MAX_OUTPUT_TOKENS),
    )


def default_settings() -> AiSettings:
    """The settings for this process, built once and reused.

    Reading the environment per call would let a stray shell variable change
    routing behaviour between two questions in the same conversation.
    """
    global _CACHED
    if _CACHED is None:
        _CACHED = settings_from_env()
    return _CACHED


_CACHED: Optional[AiSettings] = None
