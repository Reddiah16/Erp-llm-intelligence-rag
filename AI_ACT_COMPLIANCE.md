# AI System Data Handling and Regulatory Review Notes

**Last implementation review:** 2026-11-07

> **Important:** This document records observed software behavior. It is not
> legal advice, a privacy assessment, or a declaration of compliance. The
> repository review does not determine the system's legal role, risk category,
> territorial scope, or obligations under the EU AI Act, GDPR, or other laws.
> Those conclusions require review of the actual deployment, purpose, users,
> data, and operating arrangements by qualified privacy and legal reviewers.

## What Was Verified in the Repository

The findings below describe the code inspected on the date above. They do not
guarantee the settings, infrastructure, or controls of a particular deployment.

### Questions, history, and evaluation data

- The `/chat` handler keeps the raw question and answer in process memory for
  conversation history. The store is bounded to 64 sessions and 20 turns per
  session by default. A server restart clears this history; older turns and
  sessions can be evicted when those limits are reached.
- `/chat` also appends evaluation turn records to `evaluation_data/turns.jsonl`
  by default (or the configured `EVALUATION_DATA_DIR`). Those records contain
  identifiers and evaluation metadata, including a session ID and a question
  ID. For uncatalogued questions, the question ID is a deterministic hash of
  the session ID and question; the raw question text is not included in that
  turn record.
- Feedback is appended to `feedback.jsonl` in the same directory. It can
  include a free-text comment, so callers should not put personal or
  confidential information in comments unless the deployment has approved
  that processing. These JSONL records survive an API restart. No automatic
  retention or deletion schedule was found in the reviewed storage code.
- The separate legacy `POST /query` route logs the full question through the
  application logger in `src/rag/chain.py`. It also calls
  `src/evaluation/metrics.py`, which logs the first 100 characters of the
  question as an MLflow parameter, along with retrieval metrics. The MLflow
  tracking destination and its retention/access controls depend on deployment
  configuration and were not established by this source review. This is
  different from `/chat`'s evaluation JSONL behavior.
- Application, web-server, hosting, reverse-proxy, and infrastructure logs
  may have additional behavior outside the reviewed application code. Their
  contents and retention must be checked in the deployed environment.

### Documents, embeddings, and retrieval

- The RAG ingestion code chunks source documents and upserts each chunk's text,
  identifiers, and metadata into a persistent ChromaDB collection. ChromaDB
  also stores vectors. Therefore, the store does **not** contain embeddings
  and metadata only: document chunk text is persisted on disk as well.
- RAG retrieval queries that collection and returns matching chunk text and
  metadata for use in answers and citations.
- The upload endpoint accepts UTF-8 `.md` and `.txt` files. It rejects request
  bodies larger than 2,000,000 bytes, and the document helper also rejects
  decoded text larger than 2,000,000 characters. Accepted files are written
  under `documents/uploads/`, and ingestion is attempted immediately.
  Uploaded files remain on disk unless separately removed. No document
  deletion endpoint or automatic upload-retention policy was identified in
  the reviewed code.
- Which service receives document text and query text for embedding depends on
  the configured embedding provider. The configured chat model also receives
  the prompt and context needed to answer a request. Whether those services
  are local, remote, or retain request data depends on deployment
  configuration and provider terms; this review did not verify those terms.

### Structured ERP data and model configuration

- Approved analytics queries read structured ERP data from the configured
  PostgreSQL database. The source review does not establish that every
  database populated in every deployment contains only synthetic data.
- The chat and route models, embedding provider, and related service endpoints
  are configurable. The repository alone cannot establish the model build,
  hosting boundary, provider retention, or access controls used in production.
- Prompt templates are present in source files. This fact alone does not
  establish a complete model-governance, change-approval, or monitoring
  process.

## Privacy and Security Limitations

Treat user questions, feedback comments, uploaded documents, retrieved text,
and database contents as potentially personal, confidential, or otherwise
sensitive unless they have been reviewed and classified for the specific
deployment. The code review does not establish that no personal data is
collected or processed.

The reviewed implementation has multiple persistence surfaces with different
lifetimes: process-memory chat history, append-only evaluation JSONL files,
ChromaDB's persistent document text/vectors/metadata, and PostgreSQL data.
The separate `/query` route can additionally write question text to application
logs and MLflow. The repository does not establish a unified retention,
deletion, backup, access-control, or incident-response policy for those
surfaces.

This review did not verify deployment authentication/authorization, transport
encryption, filesystem/database permissions, backup protection, MLflow
configuration, cloud-provider behavior, user notices or consent, data-subject
request handling, or operational deletion procedures. Do not treat their
absence from this document as evidence that they are implemented.

## Regulatory Review Required

No EU AI Act risk classification, exemption, compliance status, or GDPR
compliance conclusion is made here. Applicability and obligations cannot be
determined from the repository alone. A qualified reviewer should assess the
actual intended and deployed use, provider/operator roles, affected people,
data flows, jurisdictions, and technical/organizational controls before making
or publishing legal claims.

Before a pilot or public deployment, document and review at least:

1. The intended purpose, users, affected people, and deployment jurisdictions.
2. The data inventory, data classification, lawful processing basis where
   relevant, notices, and provider/subprocessor data flows.
3. Retention and deletion controls for logs, MLflow, feedback, uploads,
   ChromaDB, PostgreSQL, and backups.
4. Access control, transport and at-rest protection, monitoring, and incident
   handling in the actual deployment.
5. Human oversight, output limitations, and user-facing transparency appropriate
   to the actual use case.
6. Any required legal, privacy, security, or impact assessments, as determined
   by qualified reviewers rather than this technical note.

## Review Boundary

This note supersedes the earlier unverified compliance assertions. The exact
original is preserved in [`docs/AI_ACT_COMPLIANCE.original.md`](docs/AI_ACT_COMPLIANCE.original.md)
as a backup and should not be treated as current guidance or republished as an
active compliance statement.
