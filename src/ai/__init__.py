"""The AI layer: routing, the SQL allowlist, the language-model client and the
assistant that answers a business question.

Deliberately separate from src.rag, which stays what Phase 1 built - row-record
retrieval over the cleaned CSVs - and from src.db, which stays SQL. This
package is the only place that decides which of the two answers a question.
"""
