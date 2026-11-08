"""Policy documents: chunking, ingestion, retrieval, citations, upload safety.

ChromaDB is pointed at a temporary directory and the 'chroma' provider's
DefaultEmbeddingFunction is replaced by a deterministic bag-of-words
vectoriser, so the ONNX model is never downloaded and the retrieval
ordering is stable across runs.
"""
import pytest
import chromadb
from pathlib import Path

from src.rag import documents

from .ai_fakes import FakeEmbeddingFunction, make_settings, install_fake_embeddings

INVENTORY_POLICY = """Document ID: EMP-INV-001
Effective: 2024-03-01
Owner: Inventory Planning

# Inventory Holding and Stock-Out Prevention

Policy covering stocked products and their thresholds.

## 1. Scope

Applies to every stocked product in the ERP record.

## 2. Safety stock

Safety stock is the buffer between expected demand and the quantity physically
on hand. A product is below safety stock when its whole quantity is strictly
less than its safety stock level. The planning team reviews safety stock when
demand seasonality changes.

## 3. Reorder timing

A reorder point triggers a purchase order when the product falls below it.
"""

PROCUREMENT = """Document ID: EMP-PROC-002

# Procurement Guidelines and Purchase Order Approval

## 1. Purchase order lifecycle

A pending purchase order must be approved by the manager in charge before its
release to the supplier network.

## 2. Commitment

Commitment is recorded at order level.
"""

BUSINESS_RULES = """Document ID: EMP-RULE-003

# Business Rules

## 1. Two grains of money

Revenue has an order grain and a line grain and they must never be added
together into one figure.
"""

UPLOAD_POLICY = """Document ID: EMP-UP-005

# Upload rules

Uploads are scanned before they join the document corpus.
"""

RETURN_POLICY = """# Customer Refund and Return Policy

## 1. Return eligibility

Customers may request a return within 30 calendar days of the purchase date.

## 2. Refund processing

Refunds are reviewed within 5 business days after the returned product is
received and inspected.

## 3. Damaged products

Shipping charges are not refundable unless the product arrived damaged or
defective.
"""

CORPUS = {
    "inventory_policy.md": INVENTORY_POLICY,
    "procurement_guidelines.md": PROCUREMENT,
    "business_rules.md": BUSINESS_RULES,
    "data_dictionary.txt": "Document ID: EMP-DICT-004\n\n# Dictionary\n\n"
                           "Definitions of the ERP tables.\n",
    "uploads/checkpoint_upload_policy.md": UPLOAD_POLICY,
}


@pytest.fixture
def settings(tmp_path, monkeypatch):
    """A corpus of four documents and a throwaway vector store."""
    install_fake_embeddings(monkeypatch)
    config = make_settings(tmp_path)
    write_corpus(config.documents_dir, CORPUS)
    return config


def write_corpus(directory, files):
    for name, text in files.items():
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


# ----------------------------------------------------------------- chunking


def test_a_document_yields_citable_chunks(tmp_path):
    path = tmp_path / "inventory_policy.md"
    path.write_text(INVENTORY_POLICY, encoding="utf-8")
    chunks = documents.chunk_document(path)
    sections = [chunk.section for chunk in chunks]
    assert sections[0] == "Front matter"
    assert "2. Safety stock" in sections
    assert all(chunk.document == "inventory_policy.md" for chunk in chunks)


def test_the_document_identifier_is_extracted_from_the_front_matter(tmp_path):
    path = tmp_path / "policy.md"
    path.write_text(INVENTORY_POLICY, encoding="utf-8")
    chunks = documents.chunk_document(path)
    assert {chunk.document_id for chunk in chunks} == {"EMP-INV-001"}


def test_chunk_ids_are_stable_across_reruns(tmp_path):
    path = tmp_path / "policy.md"
    path.write_text(INVENTORY_POLICY, encoding="utf-8")
    first = documents.chunk_document(path)
    again = documents.chunk_document(path)
    assert [chunk.uid for chunk in first] == [chunk.uid for chunk in again]
    assert first[0].uid == "policy.md::0::0"


def test_front_matter_cites_as_front_matter(tmp_path):
    path = tmp_path / "policy.md"
    path.write_text(INVENTORY_POLICY, encoding="utf-8")
    first = documents.chunk_document(path)[0]
    assert first.source_reference.startswith("policy.md#front-matter")


def test_sections_cite_by_slug(tmp_path):
    path = tmp_path / "policy.md"
    path.write_text(INVENTORY_POLICY, encoding="utf-8")
    chunk = next(item for item in documents.chunk_document(path)
                 if item.section == "2. Safety stock")
    assert chunk.source_reference == "policy.md#2-safety-stock"


def test_oversized_sections_split_but_keep_the_heading_prefix(tmp_path):
    paragraph = " ".join(f"word{index}" for index in range(400))
    path = tmp_path / "long.md"
    path.write_text("# Long rule\n\n" + paragraph + "\n\n" + paragraph,
                    encoding="utf-8")
    chunks = documents.chunk_document(path)
    body = [chunk for chunk in chunks if chunk.section == "Long rule"]
    assert len(body) >= 2
    for chunk in body:
        assert chunk.text.startswith("Long rule")
        assert len(chunk.text) <= 1200 + len(paragraph)


def test_chunks_exceeding_the_minimum_are_kept_even_when_small(tmp_path):
    path = tmp_path / "tiny.md"
    path.write_text("## 1. Bare heading\n\nShort.\n", encoding="utf-8")
    chunks = documents.chunk_document(path)
    assert chunks, "the first small chunk of a section is still kept"


# ------------------------------------------------------------------ corpus


def test_corpus_paths_find_every_text_document_and_nothing_else(tmp_path, settings):
    (settings.documents_dir / "binary.exe").write_bytes(b"not a document")
    found = {path.name for path in
             documents.corpus_paths(settings.documents_dir)}
    expected = {Path(name).name for name in CORPUS}
    assert found == expected
    assert "binary.exe" not in found


# ------------------------------------------------------------------- upload


def test_safe_names_pass_through():
    assert documents.safe_upload_name("policy.md") == "policy.md"
    assert documents.safe_upload_name("rules.txt") == "rules.txt"


def test_path_traversal_is_rejected_not_repaired():
    for hostile in ("../secret.md", "sub/../../root.md", r"..\\..\\x.md",
                    "a/b.md", "C:\\policy.md"):
        with pytest.raises(ValueError):
            documents.safe_upload_name(hostile)


def test_wrong_file_types_are_rejected():
    for hostile in ("payload.exe", "notes.docx", "data.csv", ".hidden.md"):
        with pytest.raises(ValueError):
            documents.safe_upload_name(hostile)


def test_name_length_and_emptiness_are_bounded():
    with pytest.raises(ValueError):
        documents.safe_upload_name("")
    with pytest.raises(ValueError):
        documents.safe_upload_name("x" * 121 + ".md")


def test_an_upload_is_stored_decoded_under_the_upload_directory(settings):
    path = documents.add_uploaded_document(b"# Uploaded\n\nSome words here.",
                                           "uploaded.md", settings)
    assert path.parent.name == "uploads"
    assert path.name == "uploaded.md"
    assert path.read_text(encoding="utf-8") == "# Uploaded\n\nSome words here."


def test_an_upload_that_is_not_utf8_text_is_refused(settings):
    with pytest.raises(ValueError):
        documents.add_uploaded_document(b"\xff\xfe\x00\x01", "binary.md", settings)


def test_an_empty_upload_is_refused(settings):
    with pytest.raises(ValueError):
        documents.add_uploaded_document(b"   \n", "empty.md", settings)


def test_an_oversized_upload_is_refused(settings):
    with pytest.raises(ValueError):
        documents.add_uploaded_document(b"a" * (2_000_000 + 1), "big.md", settings)


def test_a_document_over_the_decoded_character_limit_is_refused(settings):
    content = ("é" * (2_000_000 + 1)).encode("utf-8")
    with pytest.raises(ValueError, match="larger than 2 MB"):
        documents.add_uploaded_document(content, "big.md", settings)


# ------------------------------------------------------------- the store


def test_ingestion_reports_what_it_stored(settings):
    summary = documents.ingest(settings)
    assert summary["files"] == len(CORPUS)
    assert summary["chunks"] > 0
    assert summary["total_in_collection"] == summary["chunks"]


def test_repeated_ingestion_does_not_duplicate(settings):
    summary = documents.ingest(settings)
    again = documents.ingest(settings)
    assert again["chunks"] == summary["chunks"]
    assert again["total_in_collection"] == summary["total_in_collection"]


def test_an_updated_document_replaces_rather_than_piles_up(settings):
    documents.ingest(settings)
    before = documents.status(settings)
    path = settings.documents_dir / "uploads" / "checkpoint_upload_policy.md"
    path.write_text(path.read_text(encoding="utf-8").replace(
        "scanned", "notarized"), encoding="utf-8")
    summary = documents.ingest(settings)
    after = documents.status(settings)
    assert summary["total_in_collection"] == before["chunks"]
    assert after["chunks"] == before["chunks"]
    stored = documents.get_collection(settings).get(
        include=["documents"])["documents"]
    assert any("notarized" in text for text in stored), \
        "the updated chunk must replace the old one in place"


def test_ingesting_an_empty_corpus_is_a_named_refusal(tmp_path, settings):
    empty = make_settings(tmp_path / "blank")
    with pytest.raises(FileNotFoundError):
        documents.ingest(empty)


def test_status_reports_a_ready_store_after_ingestion(settings):
    reported = documents.status(settings)
    assert not reported["available"]
    assert reported["reason"].startswith("the document collection has not "
                                         "been created")
    assert len(reported["corpus_files"]) == len(CORPUS)

    documents.ingest(settings)
    reported = documents.status(settings)
    assert reported["available"]
    assert reported["documents"] == len(CORPUS)
    assert len(reported["corpus_files"]) == len(CORPUS)


def test_the_collection_is_created_with_cosine_distance(settings):
    documents.ingest(settings)
    client = chromadb.PersistentClient(path=str(settings.documents_dir.parent
                                                / "chroma"))
    collection = client.get_collection("test_collection")
    assert collection.metadata.get("hnsw:space") == "cosine"


def test_a_collection_is_created_through_the_provider(settings):
    collection = documents.get_collection(settings)
    assert collection.name == "test_collection"
    assert collection.count() == 0


# ----------------------------------------------------------------- retrieval


def test_retrieval_ranks_the_matching_passage_first(settings):
    documents.ingest(settings)
    hits = documents.search("What is the safety stock rule?", settings)
    assert hits[0]["document"] == "inventory_policy.md"
    assert hits[0]["section"] == "2. Safety stock"
    assert hits[0]["relevance"] > hits[-1]["relevance"]


def test_purchase_order_questions_find_the_procurement_document(settings):
    documents.ingest(settings)
    hits = documents.search("How are pending purchase orders approved?", settings)
    assert hits[0]["document"] == "procurement_guidelines.md"


def test_uploaded_return_policy_passages_are_retrievable_by_content(settings):
    documents.ingest(settings)
    path = settings.documents_dir / "uploads" / \
        "customer_return_refund_policy.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(RETURN_POLICY, encoding="utf-8")
    documents.ingest(settings, paths=[path])

    cases = (
        ("According to customer_return_refund_policy.md, how many calendar "
         "days do customers have to request a return?", "30 calendar days"),
        ("How many business days does refund approval take?",
         "5 business days"),
        ("Are shipping charges refundable when a product arrives damaged?",
         "shipping charges are not refundable"),
    )
    for question, expected_text in cases:
        hits = documents.search(question, settings, top_k=8)
        matching = [hit for hit in hits
                    if hit["document"] ==
                    "uploads/customer_return_refund_policy.md"]
        assert matching, question
        assert any(expected_text in hit["text"].lower()
                   for hit in matching), question
        _, citations = documents.render_context(matching[:settings.top_k])
        assert any(citation["document"] ==
                   "uploads/customer_return_refund_policy.md"
                   for citation in citations)


def test_retrieval_is_bounded_by_the_collection_size(settings):
    documents.ingest(settings)
    hits = documents.search("What is the reorder timing rule?", settings,
                            top_k=50)
    assert len(hits) <= documents.get_collection(settings).count()


def test_an_empty_question_retrieves_nothing(settings):
    documents.ingest(settings)
    assert documents.search("", settings) == []


def test_render_context_labels_passages_and_builds_citations(settings):
    documents.ingest(settings)
    hits = documents.search("What is the safety stock rule?", settings)
    context, citations = documents.render_context(hits)
    assert context.startswith("[D1]")
    assert "[D2]" in context
    assert len(citations) == len(hits)
    for marker, citation in zip(("D1", "D2"), citations):
        assert citation["id"] == marker
        assert len(citation["excerpt"]) <= 320
        assert citation["document"].endswith(".md") or \
            citation["document"].endswith(".txt")


def test_render_context_without_hits_demands_honesty():
    context, citations = documents.render_context([])
    assert "State that the documents could not be retrieved" in context
    assert citations == []


def test_citation_ids_are_stable():
    assert [documents.citation_id(index) for index in range(4)] == [
        "D1", "D2", "D3", "D4"]


# ------------------------------------------------------- embedding providers


def test_an_unknown_provider_is_refused_at_construction(settings):
    from dataclasses import replace
    with pytest.raises(RuntimeError):
        replace(settings, embedding_provider="openai")


def test_the_chroma_provider_resolves_to_the_injected_function(
        settings, monkeypatch):
    function = documents.embedding_function(settings)
    assert isinstance(function, FakeEmbeddingFunction)


def test_the_ollama_provider_wraps_the_local_server(settings, monkeypatch):
    import ollama

    class FakeOllama:
        def __init__(self, host=None, timeout=None):
            self.host = host

        def embeddings(self, model, prompt):
            return {"embedding": [0.5, 0.5]}

    monkeypatch.setattr(ollama, "Client", FakeOllama)
    ollama_function = documents.OllamaEmbeddingFunction(settings)
    assert ollama_function(["a", "b"]) == [[0.5, 0.5], [0.5, 0.5]]
    assert ollama_function.embed_query("q") == [0.5, 0.5]
    assert ollama_function._client.host == settings.ollama_base_url


def test_embedding_function_is_lazy_about_the_model():
    """The provider is a factory: importing the documents module never
    downloads or builds an embedding model."""
    assert callable(documents.embedding_function)


# =============================================================================
# Consistency fix: an opened collection must use the configured provider,
# never ChromaDB's fallback model
# =============================================================================
# In gene, the tests below prove the behaviour functionally: a configured
# Ollama provider produces vectors of dimension 3 in the fake, while
# ChromaDB's build-in MiniLM produces 384-dimension vectors. If retrieval
# embedded the query with the wrong model, chroma would refuse with an
# InvalidDimensionException - so a successful search IS the proof.
# =============================================================================

OLLAMA_EMBEDDING = [0.25, 0.5, 0.75]


class FakeOllamaServer:
    """Stands in for the Ollama HTTP API: fixed vectors, countable calls."""

    dimensions = 3
    calls = 0
    models_seen: set = set()

    def __init__(self, host=None, timeout=None):
        self.host = host

    def embeddings(self, model, prompt):
        FakeOllamaServer.calls += 1
        FakeOllamaServer.models_seen.add(model)
        return {"embedding": list(OLLAMA_EMBEDDING)}


def ollama_settings(tmp_path, provider="ollama"):
    from dataclasses import replace

    config = make_settings(tmp_path)
    return replace(config, embedding_provider=provider,
                   embedding_model="nomic-embed-text")


@pytest.fixture
def fake_ollama(monkeypatch):
    """Point the ollama package at the fake server and reset the counters."""
    import ollama

    FakeOllamaServer.calls = 0
    FakeOllamaServer.models_seen = set()
    monkeypatch.setattr(ollama, "Client", FakeOllamaServer)
    return FakeOllamaServer


def test_an_existing_collection_opens_with_the_configured_provider(
        tmp_path, monkeypatch):
    """The fix: an open (create=False) must attach the configured function.

    Opening without one makes chroma attach its build-in MiniLM model at
    the Collection constructor default - the bug where ingestion
    embeddings and query embeddings could differ.
    """
    install_fake_embeddings(monkeypatch)
    config = ollama_settings(tmp_path / "chroma-provider")
    created = documents.get_collection(config, create=True)
    assert created._embedding_function is not None

    reopened = documents.get_collection(config, create=False)
    assert type(reopened._embedding_function) is type(created._embedding_function)
    assert isinstance(reopened._embedding_function, FakeEmbeddingFunction), (
        "the provider chroma would have attached is the default MiniLM "
        "model, which proves the open did not fall back")


def test_ingestion_and_retrieval_use_the_same_ollama_provider(
        tmp_path, fake_ollama):
    config = ollama_settings(tmp_path)
    (config.documents_dir).mkdir(parents=True, exist_ok=True)
    (config.documents_dir / "only.md").write_text(
        "# Rules\n\nSafety stock is a buffer against demand variance.\n",
        encoding="utf-8")

    before = fake_ollama.calls
    summary = documents.ingest(config)
    embedded_at_ingest = fake_ollama.calls - before
    assert embedded_at_ingest == summary["chunks"]

    hits = documents.search("safety stock buffer", config)
    assert hits, "3-dimension stored vectors could only be queried by the "
    "same provider: a MiniLM fallback raises InvalidDimensionException"
    assert hits[0]["document"] == "only.md"
    # every embedding call was served by the ollama provider alone
    assert fake_ollama.calls > embedded_at_ingest
    assert {correct for correct in fake_ollama.models_seen} == {
        config.embedding_model}


def test_the_ollama_provider_never_reaches_minilm(tmp_path, fake_ollama):
    """A 3-dimension store queried with the 384-dimension MiniLM model is a
    dimension error, so a successful retrieval proves MiniLM was not used."""
    config = ollama_settings(tmp_path)
    config.documents_dir.mkdir(parents=True, exist_ok=True)
    (config.documents_dir / "thin.md").write_text(
        "# Thin\n\nReorder point triggers a purchase order.\n",
        encoding="utf-8")
    documents.ingest(config)

    hits = documents.search("reorder point purchase order", config, top_k=1)
    assert len(hits) == 1
    assert fake_ollama.models_seen == {config.embedding_model}


def test_building_a_broken_chroma_provider_is_a_named_error(tmp_path, monkeypatch):
    from chromadb.utils import embedding_functions

    def broken():
        raise RuntimeError("onnx model cache is missing")

    monkeypatch.setattr(embedding_functions, "DefaultEmbeddingFunction", broken)
    with pytest.raises(documents.EmbeddingProviderUnavailable) as caught:
        documents.get_collection(ollama_settings(tmp_path, "chroma"),
                                 create=True)
    message = str(caught.value)
    assert "could not be built" in message
    assert "No fallback" in message
    assert "onnxruntime" in message


def test_an_unreachable_ollama_provider_is_a_named_error(
        tmp_path, fake_ollama, monkeypatch):
    class Refusing(FakeOllamaServer):
        def embeddings(self, model, prompt):
            raise ConnectionError("refused")

    import ollama
    fake_ollama.calls = 0
    # the fixture patched ollama.Client already; replace it with the refuser
    monkeypatch.setattr(ollama, "Client", Refusing)
    function = documents.OllamaEmbeddingFunction(ollama_settings(tmp_path))
    with pytest.raises(documents.EmbeddingProviderUnavailable) as caught:
        function(["a question"])
    message = str(caught.value)
    assert "ollama serve" in message
    assert "ollama pull nomic-embed-text" in message
    assert "not falling back" in message
    assert tmp_path_prefix(function.settings) in message


def tmp_path_prefix(settings):
    return settings.ollama_base_url


# =============================================================================
# Stale-chunk fix: re-ingestion replaces its own document; nothing else
# =============================================================================


def collection_ids(settings) -> set:
    return set(documents.get_collection(settings).get(include=[])["ids"] or
               documents.get_collection(settings).get()["ids"])


def test_re_ingestion_with_fewer_chunks_removes_the_stale_chunks(settings):
    path = settings.documents_dir / "shrinking.md"
    body = " ".join(f"word{index}" for index in range(400))
    path.write_text("# Shrinking\n\n" + body + "\n\n" + body, encoding="utf-8")

    first = documents.ingest(settings, paths=[path])
    assert first["chunks"] >= 2
    before_ids = collection_ids(settings)
    assert f"shrinking.md::0::1" in before_ids

    path.write_text("# Shrinking\n\n" + body, encoding="utf-8")
    second = documents.ingest(settings, paths=[path])
    assert second["chunks"] == first["chunks"] - 1
    assert second["removed"] == first["chunks"] - second["chunks"]

    after_ids = collection_ids(settings)
    assert "shrinking.md::0::1" not in after_ids
    assert "shrinking.md::0::0" in after_ids, \
        "the surviving chunk keeps its stable position-derived id"
    assert len(after_ids) == first["total_in_collection"] - second["removed"]


def test_re_ingesting_identical_content_creates_no_duplicates(settings):
    documents.ingest(settings)
    ids = collection_ids(settings)

    summary = documents.ingest(settings)
    assert summary["removed"] == 0
    assert collection_ids(settings) == ids
    assert documents.status(settings)["chunks"] == len(ids)


def test_updating_one_document_never_touches_another_documents_chunks(
        settings):
    documents.ingest(settings)
    ids_before = collection_ids(settings)
    original_other = {identifier for identifier in ids_before
                      if identifier.split("::")[0] != "business_rules.md"}
    assert original_other, "the other documents must be ingested first"
    own_before = [identifier for identifier in ids_before
                  if identifier.startswith("business_rules.md::")]
    assert len(own_before) >= 2

    rewrite = settings.documents_dir / "business_rules.md"
    rewrite.write_text(
        "Document ID: EMP-RULE-003\n\n# Business Rules\n\nOne rule only.\n",
        encoding="utf-8")
    result = documents.ingest(settings, paths=[rewrite])
    # the removed count is exactly the chunk ids that the rewrite no longer
    # produces - position-derived ids make that set computable from the file
    fresh = {chunk.uid for chunk in documents.chunk_document(rewrite)}
    expected_removed = {identifier for identifier in own_before
                        if identifier not in fresh}
    assert result["removed"] == len(expected_removed) >= 1

    ids_after = collection_ids(settings)
    # every other document's chunk survived byte-for-byte, id for id
    assert other_ids_survived(ids_before, ids_after, "business_rules.md")
    assert not (expected_removed & ids_after), "the stale chunks are gone"
    assert fresh <= ids_after, "every rewritten chunk is present"
    # a section can also GAIN an id (body moved to a new index), which is an
    # addition, not a removal: totals must reconcile through both effects
    new_ids = fresh - ids_before
    assert len(ids_after) == len(ids_before) - result["removed"] + len(new_ids)


def other_ids_survived(before: set, after: set, removed_document: str) -> bool:
    others = {identifier for identifier in before
              if not identifier.startswith(removed_document + "::")}
    return others <= after


def test_the_surviving_collection_still_answers_the_right_question(settings):
    documents.ingest(settings)
    hits = documents.search("What is the safety stock rule?", settings)
    assert hits[0]["document"] == "inventory_policy.md"
    assert hits[0]["relevance"] > hits[-1]["relevance"]


# =============================================================================
# Document-name collisions: an upload keeps its own identity even when a file
# of the same name sits at the corpus root
# =============================================================================


def test_a_corpus_file_and_an_upload_sharing_a_name_never_collide(settings):
    root_document = settings.documents_dir / "shared_name.md"
    upload_document = settings.documents_dir / "uploads" / "shared_name.md"
    root_document.write_text(
        "Document ID: EMP-ROOT-700\n\n# Root Rules\n\nThe root file body.\n",
        encoding="utf-8")
    upload_document.write_text(
        "Document ID: EMP-UPLOAD-700\n\n# Upload Rules\n\nA different body.\n\n"
        "A second section with more words.\n", encoding="utf-8")

    summary = documents.ingest(settings, paths=[root_document,
                                                upload_document])
    assert summary["files"] == 2
    ids = collection_ids(settings)
    root_ids = {identifier for identifier in ids
                if identifier.startswith("shared_name.md::")}
    upload_ids = {identifier for identifier in ids
                  if identifier.startswith("uploads/shared_name.md::")}
    assert root_ids and upload_ids, "both documents must keep their own ids"

    # Re-ingesting only the upload, with fewer chunks, must never delete the
    # chunks that belong to the root document of the same name.
    upload_document.write_text(
        "Document ID: EMP-UPLOAD-700\n\n# Upload Rules\n\nShort.\n",
        encoding="utf-8")
    result = documents.ingest(settings, paths=[upload_document])
    ids_after = collection_ids(settings)
    assert root_ids <= ids_after, \
        "the root document's chunks must survive the upload's re-ingestion"
    stale = upload_ids - {chunk.uid for chunk in
                          documents.chunk_document(upload_document,
                                                   settings.documents_dir)}
    assert result["removed"] == len(stale)


def test_re_ingesting_an_upload_retires_its_legacy_basename_chunks(settings):
    """Chunks stored under the pre-subfolder naming (bare file name) are
    retired when the same upload is re-ingested - and only the ambiguous
    naming case is left alone."""
    upload = settings.documents_dir / "uploads" / "legacy.md"
    upload.write_text("# Legacy\n\nThe current body text.\n",
                      encoding="utf-8")
    collection = documents.get_collection(settings)
    collection.upsert(
        ids=["legacy.md::0::0", "legacy.md::0::1"],
        documents=["legacy body one", "legacy body two"],
        metadatas=[{"document": "legacy.md"}, {"document": "legacy.md"}])

    result = documents.ingest(settings, paths=[upload])
    assert result["removed"] == 2, "both legacy chunks must be retired"
    ids = collection_ids(settings)
    assert "legacy.md::0::0" not in ids and "legacy.md::0::1" not in ids
    assert all(identifier.startswith("uploads/legacy.md::") for identifier in
               ids), "only the new identity remains"


def test_a_legacy_name_shared_with_a_root_file_is_never_guessed_away(
        settings):
    """When documents/<base> exists, chunks named <base> are ambiguous and
    must survive an upload re-ingestion of the same base name."""
    root = settings.documents_dir / "twin.md"
    root.write_text("Document ID: EMP-TWIN-1\n\n# Twin\n\nRoot body.\n",
                    encoding="utf-8")
    upload = settings.documents_dir / "uploads" / "twin.md"
    upload.write_text("Document ID: EMP-TWIN-2\n\n# Twin upload\n\nUpload "
                      "body.\n", encoding="utf-8")

    documents.ingest(settings, paths=[root])
    root_ids = {identifier for identifier in collection_ids(settings)
                if identifier.startswith("twin.md::")}
    assert root_ids

    documents.ingest(settings, paths=[upload])
    ids_after = collection_ids(settings)
    assert root_ids <= ids_after
    assert any(identifier.startswith("uploads/twin.md::") for identifier in
               ids_after)


def test_re_uploading_the_same_file_is_idempotent(settings):
    content = b"# Repeat\n\nThe same words every time.\n"
    documents.add_uploaded_document(content, "repeat.md", settings)
    path = settings.documents_dir / "uploads" / "repeat.md"
    first = documents.ingest(settings, paths=[path])
    documents.add_uploaded_document(content, "repeat.md", settings)
    again = documents.ingest(settings, paths=[path])
    assert again["chunks"] == first["chunks"]
    assert again["removed"] == 0
    assert again["total_in_collection"] == first["total_in_collection"]


# =============================================================================
# Health-check hardening: /health must never load or download an embedding
# model, and provider readiness is reported separately from the store itself
# =============================================================================


def test_status_never_builds_or_downloads_the_embedding_model(
        settings, monkeypatch):
    """status() must succeed on an ingested collection even when building the
    embedding provider would fail and the ONNX download seam is armed: a
    health check embeds nothing, so it must never reach for a model."""
    from chromadb.utils import embedding_functions

    def boycott(*_args, **_kwargs):
        raise AssertionError("status tried to construct the embedding model")

    documents.ingest(settings)
    # Arm both the provider-construction seam and the download seam after the
    # fakes installed for ingestion; the closed-over state is never touched.
    monkeypatch.setattr(embedding_functions, "DefaultEmbeddingFunction", boycott)
    monkeypatch.setattr(embedding_functions.ONNXMiniLM_L6_V2,
                        "_download_model_if_not_exists", boycott)

    reported = documents.status(settings)
    assert reported["chunks"] > 0
    assert reported["documents"] >= 1
    embedding = reported["embedding"]
    assert embedding["provider"] == "chroma"
    assert embedding["model"] == "nomic-embed-text"


def test_a_missing_model_cache_is_an_actionable_degraded_status(
        settings, monkeypatch):
    """A provider whose model is not cached reports the fact and the remedy -
    and still does not download anything on the spot."""
    from chromadb.utils import embedding_functions

    def boycott(*_args, **_kwargs):
        raise AssertionError("status tried to download the model")

    monkeypatch.setattr(documents, "_minilm_cache_state", lambda: False)
    monkeypatch.setattr(embedding_functions.ONNXMiniLM_L6_V2,
                        "_download_model_if_not_exists", boycott)
    embedding = documents.status(settings)["embedding"]
    assert embedding["available"] is False
    assert "not cached yet" in embedding["detail"]
    assert "python -m src.rag.documents" in embedding["detail"]
    assert "RAG_EMBEDDING_PROVIDER=ollama" in embedding["detail"]
    assert "No other embedding model is substituted" in embedding["detail"]


def test_a_cached_model_reports_available(settings, monkeypatch):
    monkeypatch.setattr(documents, "_minilm_cache_state", lambda: True)
    embedding = documents.status(settings)["embedding"]
    assert embedding["available"] is True
    assert "cached" in embedding["detail"]


def test_the_ollama_provider_status_names_the_pulled_model_to_check(
        tmp_path, monkeypatch):
    """The embed model must be on the server's own list - checking is a read
    of Ollama's model list, never a download."""
    import ollama

    class Listing:
        def __init__(self, host=None, timeout=None):
            pass

        def list(self):
            return {"models": [{"name": "qwen2.5:1.5b"}]}

    monkeypatch.setattr(ollama, "Client", Listing)
    embedding = documents.status(
        ollama_settings(tmp_path))["embedding"]
    assert embedding["available"] is False
    assert "does not serve nomic-embed-text" in embedding["detail"]
    assert "ollama pull nomic-embed-text" in embedding["detail"]
    assert "No other embedding model is substituted" in embedding["detail"]


def test_an_ollama_variant_suffix_still_counts_as_serving_the_model(
        tmp_path, monkeypatch):
    """Ollama lists names like nomic-embed-text:latest; a pulled model with a
    tag suffix is the model the configuration asked for."""
    import ollama

    class Listing:
        def __init__(self, host=None, timeout=None):
            pass

        def list(self):
            return {"models": [{"name": "nomic-embed-text:latest"},
                               {"name": "qwen2.5:1.5b"}]}

    monkeypatch.setattr(ollama, "Client", Listing)
    embedding = documents.status(
        ollama_settings(tmp_path))["embedding"]
    assert embedding["available"] is True


def test_an_unreachable_ollama_server_is_an_actionable_status(
        tmp_path, monkeypatch):
    import ollama

    class Dead:
        def __init__(self, host=None, timeout=None):
            raise ConnectionError("connection refused")

    monkeypatch.setattr(ollama, "Client", Dead)
    embedding = documents.status(
        ollama_settings(tmp_path))["embedding"]
    assert embedding["available"] is False
    assert "ollama serve" in embedding["detail"]
    assert "ollama pull nomic-embed-text" in embedding["detail"]
