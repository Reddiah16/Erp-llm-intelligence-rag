import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from src.rag.chain import query_rag
from src.evaluation.metrics import log_query_metrics

load_dotenv()

app = FastAPI(
    title       = "ERP LLM Intelligence API",
    description = "Natural language querying of SAP-structured ERP data using RAG",
    version     = "1.0.0"
)

# Table keyword mapping for metric logging — keys map onto the canonical
# AdventureWorks table names produced by src/chunking/chunker.py
TABLE_KEYWORDS = {
    "stock":      "stock_levels",
    "safety":     "stock_levels",
    "reorder":    "stock_levels",
    "inventory":  "stock_levels",
    "material":   "product_master",
    "product":    "product_master",
    "purchase":   "purchase_orders",
    "vendor":     "purchase_orders",
    "customer":   "customer_master",
    "revenue":    "sales_orders",
    "order":      "sales_orders",
    "territory":  "sales_orders",
}


def infer_relevant_table(question: str) -> str:
    q = question.lower()
    for keyword, table in TABLE_KEYWORDS.items():
        if keyword in q:
            return table
    return "sales_orders"


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    question: str
    answer:   str
    sources:  list


@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0"}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    try:
        result = query_rag(req.question)

        # Log retrieval metrics to MLflow
        log_query_metrics(
            question       = req.question,
            sources        = result["sources"],
            relevant_table = infer_relevant_table(req.question),
            answer_length  = len(result["answer"])
        )

        return QueryResponse(
            question = result["question"],
            answer   = result["answer"],
            sources  = result["sources"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))