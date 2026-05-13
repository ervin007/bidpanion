# Bidpanion v5 — Fault-Tolerant Tender Extraction Pipeline

A robust, production-grade RAG pipeline for extracting structured data from German procurement documents, powered by **Temporal** for orchestration and **Langfuse** for observability.

## Features
- **Temporal Orchestration**: Atomic, retriable activities for ingestion, indexing, and extraction.
- **Fault Tolerance**: Automatic retries for API quotas (Vertex AI) and transient failures.
- **Hybrid Retrieval**: Combined BM25 (sparse) and ChromaDB (dense) vector search.
- **Gemini Reranking**: Context-aware reranking using Gemini 1.5 Flash.
- **Observability**: Integrated tracing and evaluation via Langfuse.
- **Containerized**: Full stack deployment using Docker Compose.

---

## Infrastructure Stack
The system consists of the following services:
- **Bidpanion Worker**: The Python executor running Temporal activities and workflows.
- **Temporal Server**: Orchestration engine for task tracking.
- **Temporal UI**: Monitoring dashboard for workflows ([http://localhost:8080](http://localhost:8080)).
- **Langfuse**: Tracing and observability dashboard ([http://localhost:3000](http://localhost:3000)).
- **PostgreSQL**: Database for Temporal and Langfuse state.
- **Redis & Clickhouse**: Performance and analytics layers for Langfuse.

---

## Setup

### 1. Prerequisites
- **Docker & Docker Compose**
- **Google Cloud Platform Account** (Vertex AI API enabled)
- **Service Account Key**: Save your GCP service account JSON key in the project root.

### 2. Configuration
Create a `.env` file in the project root:
```bash
GOOGLE_APPLICATION_CREDENTIALS="your-gcp-key.json"
LANGFUSE_PUBLIC_KEY="pk-lf-..."
LANGFUSE_SECRET_KEY="sk-lf-..."
LANGFUSE_BASE_URL="http://localhost:3000"
```

### 3. Start the Stack
```bash
docker compose up -d --build
```

---

## Usage

### Run Analysis
Place your tender document (`.txt`) in the `input/` directory, then trigger the Temporal workflow:
```bash
docker compose exec bidpanion-worker python -m temporal.run_workflow
```
*Note: The final analysis is saved to `output/result.json`.*

### Custom Inputs/Outputs
You can specify custom paths for the analysis:
```bash
docker compose exec bidpanion-worker python -m temporal.run_workflow input/my_tender.txt output/my_analysis.json
```

---

## Architecture

### Workflows
- **`TenderExtractionWorkflow`**: The parent orchestrator. It prepares document indices once and spawns parallel child workflows for each field.
- **`FieldExtractionWorkflow`**: A per-field workflow that manages the retrieval, reranking, and extraction steps for a single "question".

### Activities
- **`prepare_indices`**: Chunks the document and builds BM25/Vectorstore indices (cached in memory).
- **`extract_field_activity`**: Performs hybrid search, LLM reranking, and structured data extraction.
- **`save_final_results`**: Aggregates all field outputs into the final schema and performs validation.

---

## Monitoring & Debugging
- **Temporal Dashboard**: [http://localhost:8080](http://localhost:8080) - View real-time status, retries, and history of every extraction task.
- **Langfuse Traces**: [http://localhost:3000](http://localhost:3000) - Inspect the LLM prompts, retrieval results, and performance metrics.
- **Worker Logs**: `docker compose logs -f bidpanion-worker`
