import os
import chromadb
from pathlib import Path
from dotenv import load_dotenv
import logging
from langchain_ollama import OllamaEmbeddings
from src.chunking.chunker import load_and_chunk_all

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")
log = logging.getLogger(__name__)

CHROMA_PATH     = os.getenv("CHROMA_PATH", "chroma_db")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL     = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
COLLECTION_NAME = "erp_data"


def get_chroma_client():
    return chromadb.PersistentClient(path=CHROMA_PATH)


def ingest():
    log.info(f"Using embedding model: {EMBED_MODEL}")
    log.info(f"ChromaDB path: {CHROMA_PATH}")

    chunks = load_and_chunk_all()

    client     = get_chroma_client()
    collection = client.get_or_create_collection(
        name     = COLLECTION_NAME,
        metadata = {"hnsw:space": "cosine"}
    )

    # Check existing count
    existing = collection.count()
    log.info(f"Existing documents in ChromaDB: {existing}")

    if existing >= len(chunks):
        log.info("ChromaDB already up to date — skipping ingestion.")
        return

    log.info(f"Embedding {len(chunks)} chunks with {EMBED_MODEL}...")

    embedder = OllamaEmbeddings(
        model    = EMBED_MODEL,
        base_url = OLLAMA_BASE_URL
    )

    # Batch embed for efficiency
    texts     = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]
    ids       = [f"chunk_{i}" for i in range(len(chunks))]

    batch_size = 50
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_meta  = metadatas[i:i + batch_size]
        batch_ids   = ids[i:i + batch_size]

        embeddings = embedder.embed_documents(batch_texts)

        collection.add(
            documents  = batch_texts,
            embeddings = embeddings,
            metadatas  = batch_meta,
            ids        = batch_ids
        )
        log.info(f"  Ingested batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")

    log.info(f"Ingestion complete — {collection.count()} documents in ChromaDB")


if __name__ == "__main__":
    ingest()