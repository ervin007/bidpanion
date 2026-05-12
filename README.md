# Bidpanion v3 — Tender Extraction Pipeline

A high-fidelity RAG pipeline for extracting structured data from German procurement documents.

## Prerequisites
- **uv**: Fast Python package manager ([installation](https://github.com/astral-sh/uv))
- **Docker**: For running the Langfuse observability stack.

---

## Setup

### 1. Infrastructure (Langfuse)
Start the tracing stack (Postgres, Redis, Clickhouse, Langfuse):
```bash
docker compose up -d
```
Access the dashboard at `http://localhost:3000`.

### 2. Dependencies
Install and sync dependencies:
```bash
uv sync
```

### 3. Environment
Create a `.env` file with your credentials:
```bash
GOOGLE_APPLICATION_CREDENTIALS="credentials.json"
LANGFUSE_PUBLIC_KEY="pk-lf-..."
LANGFUSE_SECRET_KEY="sk-lf-..."
LANGFUSE_HOST="http://localhost:3000"
```

---

## Usage

### Run Extraction
Place your `.txt` tender document in `input/` and run:
```bash
uv run main.py
```
*Note: Result is saved to `output/result.json`.*

### Run Evaluation
```bash
uv run evaluate.py
```

---

## Tinkering & Iteration

Most logic resides in **`config.py`**. Iterate there to improve results:

- **Prompts**: Edit `field["instruction"]` to change extraction behavior.
- **Queries**: Edit `field["queries"]` to improve retrieval recall.
- **Global Rules**: Edit `ROUTING_RULES` for cross-field disambiguation.
- **Tuning**: Adjust `TOP_K_DENSE`, `TOP_K_RERANK`, or `CHUNK_SIZE` for performance/cost.

For technical details on the architecture, see [instructions.md](file:///Users/ervinshaqiri/Documents/Projects/Bidpanion%20v3/instructions.md).
