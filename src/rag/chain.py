import os
import chromadb
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from src.llm.prompts import ERP_RAG_PROMPT
import logging

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")
log = logging.getLogger(__name__)

CHROMA_PATH     = os.getenv("CHROMA_PATH", "chroma_db")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "llama3.2")
EMBED_MODEL     = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
COLLECTION_NAME = "erp_data"
TOP_K           = 8


def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_collection(COLLECTION_NAME)


def retrieve(query: str, collection, top_k: int = TOP_K) -> list:
    embedder   = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)
    query_emb  = embedder.embed_query(query)
    results    = collection.query(
        query_embeddings = [query_emb],
        n_results        = top_k,
        include          = ["documents", "metadatas", "distances"]
    )
    docs = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        docs.append({
            "text":       doc,
            "metadata":   meta,
            "distance":   round(dist, 4),
            "relevance":  round(1 - dist, 4)
        })
    return docs


def build_context(docs: list) -> str:
    return "\n\n".join([d["text"] for d in docs])


def query_rag(question: str) -> dict:
    log.info(f"Query: {question}")

    collection = get_collection()
    docs       = retrieve(question, collection)
    context    = build_context(docs)

    llm    = OllamaLLM(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
    chain  = ERP_RAG_PROMPT | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})

    return {
        "question": question,
        "answer":   answer,
        "sources":  [
            {
                "record_id": d["metadata"].get("record_id", ""),
                "table":     d["metadata"].get("table", ""),
                "relevance": d["relevance"],
                "text":      d["text"]
            }
            for d in docs
        ]
    }


if __name__ == "__main__":
    test_questions = [
        "Which materials are below safety stock?",
        "What is the total revenue from sales orders?",
        "Show me the top customers by order value.",
    ]
    for q in test_questions:
        print(f"\n{'='*60}")
        print(f"Q: {q}")
        result = query_rag(q)
        print(f"A: {result['answer']}")
        print(f"Sources used: {len(result['sources'])}")