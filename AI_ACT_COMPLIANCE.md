# EU AI Act & GDPR Compliance Notes

## Risk Classification
This system provides AI-generated responses based on retrieved ERP data.
It is intended as a decision-support tool, not an autonomous decision-maker.
Under the EU AI Act, this system is classified as limited risk.

## Transparency Measures
- Users are informed they are interacting with an AI system
- Every answer includes retrieved source records so users can verify responses
- Retrieval quality metrics (MRR, Recall@5) are tracked via MLflow
- Prompt templates are versioned in src/llm/prompts.py

## Data Handling
- The system uses the AdventureWorks dataset — a publicly available Microsoft
  sample database containing synthetic business data
- AdventureWorks contains anonymised customer IDs and order records designed
  for demonstration purposes — it does not contain real personal data
- ChromaDB stores only document embeddings and metadata — no raw text is
  persisted beyond the ingestion process

## Model Governance
- Retrieval quality metrics (MRR, Recall@5) are tracked via MLflow
- Prompt templates are versioned in source control
- LLM model version is configurable via environment variables

## GDPR Measures
- No personal data is collected from users of this system
- AdventureWorks data does not contain real personally identifiable information
- Query inputs are not logged or stored beyond the current session
- ChromaDB embeddings represent document content only, not user interactions
