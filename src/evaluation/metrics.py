import mlflow
import logging
from typing import List, Dict

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")
log = logging.getLogger(__name__)


def reciprocal_rank(sources: List[Dict], relevant_table: str) -> float:
    """
    Mean Reciprocal Rank — checks if the most relevant table
    appears in the top results and at what position.
    """
    for i, source in enumerate(sources):
        if source.get("table") == relevant_table:
            return 1.0 / (i + 1)
    return 0.0


def recall_at_k(sources: List[Dict], relevant_table: str, k: int = 5) -> float:
    """
    Recall@K — checks if at least one relevant record
    appears in the top K retrieved sources.
    """
    top_k = sources[:k]
    tables = [s.get("table") for s in top_k]
    return 1.0 if relevant_table in tables else 0.0


def log_query_metrics(
    question:       str,
    sources:        List[Dict],
    relevant_table: str,
    answer_length:  int
):
    """Log retrieval metrics to MLflow."""
    try:
        mlflow.set_experiment("erp-rag-evaluation")

        with mlflow.start_run(run_name="rag_query"):
            rr      = reciprocal_rank(sources, relevant_table)
            recall5 = recall_at_k(sources, relevant_table, k=5)
            avg_rel = sum(s.get("relevance", 0) for s in sources) / len(sources) if sources else 0

            mlflow.log_param("question",       question[:100])
            mlflow.log_param("relevant_table", relevant_table)
            mlflow.log_metric("reciprocal_rank",    rr)
            mlflow.log_metric("recall_at_5",        recall5)
            mlflow.log_metric("avg_relevance",      round(avg_rel, 4))
            mlflow.log_metric("num_sources",        len(sources))
            mlflow.log_metric("answer_length_chars", answer_length)

            log.info(f"Metrics logged — RR: {rr:.3f} | Recall@5: {recall5:.3f} | Avg relevance: {avg_rel:.3f}")

    except Exception as e:
        log.warning(f"MLflow logging failed: {e}")


if __name__ == "__main__":
    # Smoke test using canonical AdventureWorks table names and ProductIDs
    sample_sources = [
        {"table": "stock_levels", "record_id": "776", "relevance": 0.85},
        {"table": "stock_levels", "record_id": "777", "relevance": 0.76},
        {"table": "product_master", "record_id": "778", "relevance": 0.71},
    ]
    log_query_metrics(
        question       = "Which products are below safety stock level?",
        sources        = sample_sources,
        relevant_table = "stock_levels",
        answer_length  = 250
    )
    print("Metrics logged successfully.")