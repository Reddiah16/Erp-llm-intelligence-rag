"""Unstructured business documents: chunking, ingestion, retrieval, citations.

This is the RAG side of Phase 3, and it is deliberately NOT the Phase 1
pipeline. src/rag/chain.py retrieves ERP ROW RECORDS rendered from the cleaned
CSVs; that is a different corpus with a different purpose, and mixing row
records into the same collection would let a policy question return a customer
address and a revenue question return a paragraph of prose. Separate
collections keep the two honest.

Only documents are ingested here, and the documents in documents/ contain rules
and definitions rather than business figures. That is what makes the brief's
"do not use RAG to calculate numerical metrics" a structural property rather
than a prompt instruction: the corpus holds nothing to calculate with, so a
number in an answer can only have come from SQL.
"""
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.ai.settings import AiSettings, default_settings

TEXT_SUFFIXES = (".md", ".txt")
UPLOAD_DIR_NAME = "uploads"

MAX_CHUNK_CHARS = 1200
MIN_CHUNK_CHARS = 40


class EmbeddingProviderUnavailable(RuntimeError):
    """The configured embedding provider could not be reached or built.

    Raised instead of letting a lower-level error surface as a bare
    stack trace or, worse, as a quiet fallback to a different embedding
    model - a collection opened under another model's vectors is not an
    error, it is noise, and a wrong retrieval with plausible-looking text
    is worse than no retrieval at all.
    """

# "## 2. Safety stock" is a citable unit, so the section heading is the first
# thing the reader of a citation sees.
_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_DOCUMENT_ID = re.compile(r"^\s*Document ID:\s*(\S+)", re.MULTILINE)
_VERSION = re.compile(r"^\s*(Effective|Version|Last reviewed):", re.MULTILINE | re.IGNORECASE)


def slug(value: str, limit: int = 48) -> str:
    """A heading as a stable, readable reference fragment."""
    lowered = value.lower()
    replaced = re.sub(r"[^a-z0-9]+", "-", lowered).strip("-")
    return replaced[:limit].rstrip("-") or "section"


@dataclass(frozen=True)
class DocumentChunk:
    text: str
    document: str
    section: str
    section_index: int
    chunk_index: int
    source_reference: str
    document_id: str = ""

    @property
    def uid(self) -> str:
        """Stable across re-ingestion, so an update replaces rather than piles up.

        Ids derived from the file name and position mean running ingestion twice
        cannot double the corpus - the failure mode that makes a retrieval score
        climb for reasons that have nothing to do with relevance.
        """
        return f"{self.document}::{self.section_index}::{self.chunk_index}"

    def metadata(self) -> Dict[str, Any]:
        return {
            "document": self.document,
            "document_id": self.document_id,
            "section": self.section,
            "section_index": self.section_index,
            "chunk_index": self.chunk_index,
            "source_reference": self.source_reference,
            "characters": len(self.text),
            "kind": "policy_document",
        }


# --------------------------------------------------------------------- parsing

def _sections(text: str) -> List[Tuple[str, List[str]]]:
    """Split a document into (heading, body lines) blocks.

    Lines before the first heading become a "Front matter" block: they carry the
    document ID, owner and effective date, which are exactly what a citation
    should be able to name.
    """
    blocks: List[Tuple[str, List[str]]] = [("Front matter", [])]
    for line in text.splitlines():
        match = _HEADING.match(line)
        if match:
            blocks.append((match.group(2).strip(), []))
        else:
            blocks[-1][1].append(line)
    if not "".join(blocks[0][1]).strip():
        blocks.pop(0)
    return blocks


def _paragraphs(lines: Iterable[str]) -> List[str]:
    """Non-empty paragraph groups, in order."""
    paragraphs, current = [], []
    for line in lines:
        if line.strip():
            current.append(line.rstrip())
        elif current:
            paragraphs.append("\n".join(current))
            current = []
    if current:
        paragraphs.append("\n".join(current))
    return paragraphs


def _fit(heading: str, paragraphs: List[str],
         limit: int = MAX_CHUNK_CHARS) -> List[str]:
    """Group paragraphs so no chunk exceeds the limit unless one paragraph does.

    A chunk is prefixed with its heading, because an embedding of "2. 3 5" from
    the middle of a table is useless for retrieval and misleading as a citation.
    A single paragraph longer than the limit is kept whole rather than cut
    mid-sentence: a policy rule split across two chunks reads as two partial
    rules, which is worse than one oversized chunk.
    """
    prefix = f"{heading}\n" if heading else ""
    chunks: List[str] = []
    current = ""
    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}" if current else paragraph
        if current and len(prefix) + len(candidate) > limit:
            chunks.append(current)
            current = paragraph
        else:
            current = candidate
    if current:
        chunks.append(current)
    return [prefix + chunk for chunk in chunks]


def chunk_document(path: Path, root: Optional[Path] = None) -> List[DocumentChunk]:
    """One file, as citable chunks.

    `root` makes the document identity the path relative to it (`uploads/x.md`
    against `x.md`), which is what keeps a same-named file in a subfolder from
    colliding with a file at the root: chunk ids and the `document` metadata are
    both derived from this name, so two different files must never reduce to one.
    Without a root the bare file name is used, which is what standalone callers
    and every chunk-level test rely on.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    identifier = _DOCUMENT_ID.search(text)
    document_id = identifier.group(1) if identifier else ""
    if root is not None:
        try:
            name = path.relative_to(root).as_posix()
        except ValueError:
            name = path.name
    else:
        name = path.name
    chunks: List[DocumentChunk] = []
    for section_index, (heading, lines) in enumerate(_sections(text)):
        fitted = _fit(heading, _paragraphs(lines))
        if not fitted:
            continue
        reference = (f"{name}#front-matter" if heading == "Front matter"
                     else f"{name}#{slug(heading)}")
        for chunk_index, chunk in enumerate(fitted):
            if len(chunk.strip()) < MIN_CHUNK_CHARS and chunk_index:
                continue
            chunks.append(DocumentChunk(
                text=chunk, document=name, section=heading,
                section_index=section_index, chunk_index=chunk_index,
                source_reference=reference + (f":{chunk_index + 1}"
                                              if len(fitted) > 1 else ""),
                document_id=document_id))
    return chunks


def corpus_paths(documents_dir: Path) -> List[Path]:
    """Every text document in the corpus, uploads included, name-ordered."""
    if not documents_dir.exists():
        return []
    return sorted(p for p in documents_dir.rglob("*")
                  if p.is_file() and p.suffix.lower() in TEXT_SUFFIXES)


def chunk_corpus(documents_dir: Path) -> List[DocumentChunk]:
    chunks: List[DocumentChunk] = []
    for path in corpus_paths(documents_dir):
        chunks.extend(chunk_document(path, documents_dir))
    return chunks


# ---------------------------------------------------------------------- store

def _build_embedding_function(settings: AiSettings):
    """The provider for this configuration, or an actionable failure.

    Building the provider is where a configuration problem shows up: an
    unreachable Ollama server or a missing ONNX model should be reported
    with its remedy, and never answered by choosing a different model
    instead - the vectors already in the store would dispute everything
    the replacement model says.
    """
    try:
        return embedding_function(settings)
    except EmbeddingProviderUnavailable:
        raise
    except Exception as exc:                 # noqa: BLE001 - made actionable
        detail = " ".join(str(exc).split())[:200] or type(exc).__name__
        if settings.embedding_provider == "ollama":
            remedy = (f"Start Ollama with 'ollama serve' at "
                      f"{settings.ollama_base_url} and run 'ollama pull "
                      f"{settings.embedding_model}'.")
        else:
            remedy = ("The bundled MiniLM model is pinned with chromadb in "
                      "requirements: reinstall the pinned onnxruntime and "
                      "clear the model cache that is listed in the error "
                      "above.")
        raise EmbeddingProviderUnavailable(
            f"the configured embedding provider "
            f"({settings.embedding_provider}) could not be built: {detail}. "
            f"{remedy} No fallback embedding model is used, because mixing "
            f"two embedding models in one collection makes retrieval "
            f"meaningless.") from exc


def get_collection(settings: Optional[AiSettings] = None,
                   create: bool = True):
    """The document collection, with its embedding function attached.

    The SAME function is attached on creation and on every open. Chroma
    only keeps an embedding function on the collection object it returned
    when it was created; a collection opened without one silently falls
    back to ChromaDB's build-in model, and a query embedded by a
    different model than the stored vectors is not an error - it is
    worse, it is noise. Opening without the configured provider is
    therefore not allowed here.
    """
    import chromadb

    settings = settings or default_settings()
    path = settings.anchor(settings.chroma_path)
    path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(path))
    arguments: Dict[str, Any] = {
        "name": settings.document_collection,
        "embedding_function": _build_embedding_function(settings),
    }
    if create:
        return client.get_or_create_collection(
            metadata={"hnsw:space": "cosine"}, **arguments)
    return client.get_collection(**arguments)


def embedding_function(settings: AiSettings):
    """The embedder for this configuration, built only when one is needed.

    ChromaDB's default embedding function downloads an ONNX model when it is
    constructed. Building it at import time would mean that importing the API
    - or calling GET /health - reached for the network, so this is a function
    and every caller that does not embed never calls it.
    """
    if settings.embedding_provider == "ollama":
        return OllamaEmbeddingFunction(settings)

    from chromadb.utils import embedding_functions
    return embedding_functions.DefaultEmbeddingFunction()


class OllamaEmbeddingFunction:
    """Chroma's embedding-function protocol over the local Ollama server."""

    def __init__(self, settings: AiSettings) -> None:
        self.settings = settings
        self._client = None

    def _ollama(self):
        if self._client is None:
            import ollama
            self._client = ollama.Client(host=self.settings.ollama_base_url,
                                         timeout=self.settings.llm_timeout)
        return self._client

    def _unavailable(self, exc: Exception) -> EmbeddingProviderUnavailable:
        base_url = self.settings.ollama_base_url
        model = self.settings.embedding_model
        return EmbeddingProviderUnavailable(
            f"the Ollama embedding provider at {base_url} could not embed "
            f"input ({type(exc).__name__}). Start the server with 'ollama "
            f"serve' and pull the model with 'ollama pull {model}'. The "
            f"document store is not falling back to another embedding "
            f"model, because vectors from two different models cannot be "
            f"compared.")

    def __call__(self, input: List[str]) -> List[List[float]]:  # noqa: A002
        try:
            return [self._ollama().embeddings(model=self.settings.embedding_model,
                                              prompt=text)["embedding"]
                    for text in input]
        except EmbeddingProviderUnavailable:
            raise
        except Exception as exc:             # noqa: BLE001 - made actionable
            raise self._unavailable(exc) from exc

    def embed_documents(self, input: List[str]) -> List[List[float]]:
        return self(input)

    def embed_query(self, input: str) -> List[float]:
        return self([input])[0]


def embedding_provider_status(settings: AiSettings) -> Dict[str, Any]:
    """Whether the configured embedding provider can embed right now - WITHOUT
    loading or downloading any model.

    This is deliberately the lightest possible probe. Building the embedding
    function is cheap but it is the first step on the road to an embedding
    call, and a health check must never pull a model over the network or
    react to an absent one by picking another. The result names the provider
    and model, says whether they are reachable, and carries the remedy when
    they are not, so /health can report a degraded document component that
    a person can act on.
    """
    info: Dict[str, Any] = {
        "provider": settings.embedding_provider,
        "model": settings.embedding_model,
        "available": False,
        "detail": "",
    }
    try:
        cached = _minilm_cache_state() \
            if settings.embedding_provider != "ollama" else None
        if settings.embedding_provider == "ollama":
            import ollama
            listing = ollama.Client(
                host=settings.ollama_base_url, timeout=10).list()
            names = [model.get("name") or model.get("model", "")
                     for model in listing.get("models", [])]
            wanted = settings.embedding_model
            if any(name == wanted or name.startswith(f"{wanted}:")
                   for name in names):
                info["available"] = True
                info["detail"] = f"{wanted} is served by Ollama " \
                                 f"at {settings.ollama_base_url}"
            else:
                info["detail"] = (
                    f"Ollama at {settings.ollama_base_url} is reachable but "
                    f"does not serve {wanted}. Pull it with 'ollama pull "
                    f"{wanted}'. No other embedding model is substituted, "
                    f"so the document store stays unavailable for "
                    f"ingestion and retrieval until then.")
        elif cached:
            info["available"] = True
            info["detail"] = ("the bundled MiniLM model is already cached "
                              "locally")
        elif cached is None:
            info["detail"] = (
                "the bundled MiniLM provider could not be checked against "
                "its cache (chromadb internals changed). Health checks "
                "still never download it; the first ingestion or retrieval "
                "that actually embeds will acquire the model if it is "
                "missing, and no other model is substituted.")
        else:
            info["detail"] = (
                "the bundled MiniLM model is not cached yet; it is "
                "downloaded once, by the first ingestion or retrieval that "
                "actually embeds - never by /health. Run 'python -m "
                "src.rag.documents' when you are ready to fetch it, or set "
                f"RAG_EMBEDDING_PROVIDER=ollama with a pulled "
                f"{settings.embedding_model} to avoid the download "
                "entirely. No other embedding model is substituted.")
    except Exception as exc:  # noqa: BLE001 - reported, health stays up
        detail = " ".join(str(exc).split())[:200] or type(exc).__name__
        if settings.embedding_provider == "ollama":
            info["detail"] = (
                f"the Ollama server at {settings.ollama_base_url} could not "
                f"be reached ({detail}). Start it with 'ollama serve' and "
                f"pull the model with 'ollama pull "
                f"{settings.embedding_model}'. No other embedding model is "
                "substituted.")
        else:
            info["detail"] = (
                f"the bundled MiniLM provider could not be checked "
                f"({detail}). Reinstall the pinned onnxruntime from the "
                "requirements files and retry.")
    return info


def _minilm_cache_state() -> Optional[bool]:
    """True/false whether the ONNX MiniLM files are cached; None if unknown.

    Reads chromadb's own download path without importing onnxruntime beyond
    what chromadb already imports and without touching the network. The
    bundled model's constructor does not download (the download happens on
    the first actual embed call), but this check retrieves the answer cheaply
    so a health report can say where things stand instead of saying the
    provider is fine and discovering a 100 MB download during the next
    ingestion.
    """
    try:
        from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
        folder = (Path(ONNXMiniLM_L6_V2.DOWNLOAD_PATH)
                  / ONNXMiniLM_L6_V2.EXTRACTED_FOLDER_NAME)
        return ((folder / "model.onnx").exists()
                and (folder / "tokenizer.json").exists())
    except Exception:  # noqa: BLE001 - unknown is not "fine"
        return None


def status(settings: Optional[AiSettings] = None) -> Dict[str, Any]:
    """What the document store can answer right now, without embedding anything.

    `available` is false when the collection does not exist or holds no
    documents, and the reason says which - the assistant must not let a policy
    question silently return an empty citation list.

    The collection is opened WITHOUT an embedding function: counting and
    metadata reads never embed, so a health check can neither download a
    model nor depend on the provider being loadable. Provider readiness is
    reported separately in the `embedding` block, where a missing or
    unreachable model is a degraded status with a remedy rather than a
    surprise download.
    """
    settings = settings or default_settings()
    documents_dir = settings.anchor(settings.documents_dir)
    corpus = corpus_paths(documents_dir)
    result: Dict[str, Any] = {
        "collection": settings.document_collection,
        "chroma_path": str(settings.anchor(settings.chroma_path)),
        "documents_dir": str(documents_dir),
        "corpus_files": [p.name for p in corpus],
        "available": False,
        "chunks": 0,
        "documents": 0,
        "reason": "",
        "embedding": embedding_provider_status(settings),
    }
    try:
        import chromadb
        path = settings.anchor(settings.chroma_path)
        path.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(path))
        # No embedding_function is passed, so nothing is constructed beyond
        # what chromadb built at import time, and no query_texts is ever
        # issued here.
        collection = client.get_collection(name=settings.document_collection)
    except Exception as exc:  # noqa: BLE001 - reported as unavailable
        details = " ".join(str(exc).split())[:160]
        result["reason"] = (f"the document collection has not been created "
                            f"({details}); run 'python -m src.rag.documents' "
                            f"to ingest {len(corpus)} files")
        return result
    chunks = collection.count()
    result.update(available=chunks > 0, chunks=chunks)
    if chunks:
        result["documents"] = len({
            row.get("document")
            for row in collection.get(include=["metadatas"])["metadatas"]})
    else:
        result["reason"] = "the collection exists but holds no chunks"
    return result


def ingest(settings: Optional[AiSettings] = None,
           paths: Optional[List[Path]] = None) -> Dict[str, Any]:
    """Chunk and upsert the corpus. Idempotent by construction.

    Upsert with position-derived ids means a second run replaces the same rows
    rather than adding duplicates, so re-ingesting after an edit cannot inflate
    the corpus and make retrieval look better than it is.

    When one pass ingests more than one document, each document's surplus
    chunks are computed separately: a document edited down to fewer chunks
    leaves its own old chunks behind, but a chunk from another document is
    never removed by that bookkeeping.
    """
    settings = settings or default_settings()
    documents_dir = settings.anchor(settings.documents_dir)
    paths = paths if paths is not None else corpus_paths(documents_dir)

    chunks: List[DocumentChunk] = []
    for path in paths:
        chunks.extend(chunk_document(path, documents_dir))
    if not chunks:
        raise FileNotFoundError(
            f"no .md or .txt documents found under {documents_dir}")

    fresh_ids = {chunk.uid for chunk in chunks}
    collection = get_collection(settings)
    aliases = {chunk.document: _legacy_aliases(chunk.document, documents_dir)
               for chunk in chunks}
    stale_ids = _surplus_ids(collection, chunks, fresh_ids, aliases)

    collection.upsert(
        ids=[chunk.uid for chunk in chunks],
        documents=[chunk.text for chunk in chunks],
        metadatas=[chunk.metadata() for chunk in chunks],
    )
    if stale_ids:
        collection.delete(ids=stale_ids)
    return {
        "files": len(paths),
        "chunks": len(chunks),
        "removed": len(stale_ids),
        "collection": settings.document_collection,
        "total_in_collection": collection.count(),
    }


def _legacy_aliases(document: str, documents_dir: Path) -> List[str]:
    """Older identity forms of the same file, if any are unambiguous.

    Chunks ingested before uploads carried a subfolder-prefixed identity were
    stored under the bare file name. Retiring those on re-ingestion keeps an
    upgrade of the store from leaving unsearchable duplicates behind - but the
    bare name is only safe to retire when no corpus-root file owns it: when
    documents/<base> exists, a legacy <base> chunk set is ambiguous and must be
    left alone rather than guessed away.
    """
    base = Path(document).name
    if document == base:
        return []
    if (documents_dir / base).exists():
        return []
    return [base]


def _surplus_ids(collection, chunks: List[DocumentChunk],
                 fresh_ids: set,
                 aliases: Optional[Dict[str, List[str]]] = None) -> List[str]:
    """Old chunk ids of exactly the documents being re-ingested.

    One lookup per touched document, scoped by the `document` metadata the
    chunker writes, so a document edited down to fewer chunks loses its own
    leftovers while every other document's chunks stay untouched. The ids
    the new pass produces itself are excluded: position-derived ids are
    stable, and an upserted id must not be deleted after the upsert.
    """
    aliases = aliases or {}
    surplus: List[str] = []
    touched = sorted({chunk.document for chunk in chunks})
    for document in touched:
        candidates = [document, *aliases.get(document, [])]
        if len(candidates) == 1:
            found = collection.get(where={"document": document},
                                   include=["metadatas"])
        else:
            found = collection.get(where={"document": {"$in": candidates}},
                                   include=["metadatas"])
        surplus.extend(
            identifier for identifier in found["ids"]
            if identifier not in fresh_ids)
    return surplus


def search(question: str, settings: Optional[AiSettings] = None,
           top_k: Optional[int] = None) -> List[Dict[str, Any]]:
    """Retrieve policy passages, best first, each with a citable reference.

    Cosine distance becomes relevance the same way src/rag/chain.py reports it,
    so both retrievals mean the same number to a reader.
    """
    settings = settings or default_settings()
    text = (question or "").strip()
    if not text:
        return []
    collection = get_collection(settings, create=False)
    if collection.count() == 0:
        return []
    found = collection.query(
        query_texts=[text],
        n_results=min(top_k or settings.top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    hits: List[Dict[str, Any]] = []
    for passage, meta, distance in zip(found["documents"][0],
                                       found["metadatas"][0],
                                       found["distances"][0]):
        hits.append({
            "document": meta.get("document", ""),
            "document_id": meta.get("document_id", ""),
            "section": meta.get("section", ""),
            "reference": meta.get("source_reference", ""),
            "relevance": round(1 - distance, 4),
            "distance": round(distance, 4),
            "text": passage,
        })
    return hits


def citation_id(index: int) -> str:
    """The marker an answer cites: [D1], [D2]... one per retrieved passage."""
    return f"D{index + 1}"


def render_context(hits: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    """The block the model reads, plus the citation list shown to the user.

    Each passage is labelled with the document, the section and the citation
    marker before its text, so a model quoting policy has something to point at
    and a reader can check it.
    """
    if not hits:
        return ("No policy documents are available for this question. State "
                "that the documents could not be retrieved instead of "
                "describing policy."), []
    blocks, citations = [], []
    for index, hit in enumerate(hits):
        marker = citation_id(index)
        blocks.append(
            f"[{marker}] {hit['document']} - {hit['section']} "
            f"(relevance {hit['relevance']}):\n{hit['text']}")
        citations.append({
            "id": marker,
            "document": hit["document"],
            "document_id": hit["document_id"],
            "section": hit["section"],
            "reference": hit["reference"],
            "relevance": hit["relevance"],
            "excerpt": hit["text"][:320],
        })
    return "\n\n".join(blocks), citations


def safe_upload_name(filename: str) -> str:
    """A file name that cannot escape the upload directory.

    Rejects anything path-shaped or with an unexpected suffix rather than
    sanitising it into something quiet: "../../etc/passwd" and "payload.exe"
    both become an error the caller can read.
    """
    tail = Path(filename or "").name
    if not tail:
        raise ValueError("the uploaded file has no name")
    if tail != filename:
        raise ValueError("the uploaded file name may not contain a path")
    if tail.startswith("."):
        raise ValueError("the uploaded file name may not be hidden or relative")
    if Path(tail).suffix.lower() not in TEXT_SUFFIXES:
        raise ValueError(f"only {', '.join(TEXT_SUFFIXES)} documents are "
                         f"accepted, not {Path(tail).suffix!r}")
    if len(tail) > 120:
        raise ValueError("the uploaded file name is too long")
    return tail


def add_uploaded_document(content: bytes, filename: str,
                          settings: Optional[AiSettings] = None) -> Path:
    """Store one uploaded document and return its path.

    Decoded as UTF-8 before writing, so a binary body never lands in the corpus
    wearing a text suffix.
    """
    settings = settings or default_settings()
    name = safe_upload_name(filename)
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("the uploaded file is not UTF-8 text") from exc
    if not text.strip():
        raise ValueError("the uploaded file is empty")
    if len(text) > 2_000_000:
        raise ValueError("the uploaded document is larger than 2 MB")

    directory = settings.anchor(settings.documents_dir) / UPLOAD_DIR_NAME
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(text, encoding="utf-8")
    return path


def main() -> int:
    settings = default_settings()
    summary = ingest(settings)
    reported = status(settings)
    print(f"Ingested {summary['chunks']} chunks from {summary['files']} files "
          f"into {summary['collection']}"
          f" ({reported['documents']} documents, "
          f"{reported['chunks']} chunks total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
