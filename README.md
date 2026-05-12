# Bidpanion v3 — Tender Extraction Pipeline

A high-fidelity RAG (Retrieval-Augmented Generation) pipeline designed to extract structured data from complex German procurement (tender) documents.

## Workflow Overview

The pipeline follows a multi-stage process to ensure maximum recall and precision:

1.  **Ingestion**: Loads the target document from `input/` and splits it into manageable chunks.
2.  **Indexing**: Simultaneously builds a **BM25 index** (for keyword-heavy retrieval) and a **ChromaDB Vectorstore** (for semantic retrieval).
3.  **Extraction Loop**: For every field defined in the schema:
    *   **Hybrid Retrieval**: Executes multiple queries (keyword + semantic) to find relevant sections.
    *   **Reranking**: Uses Gemini to analyze and rank the most relevant chunks from the retrieved set.
    *   **Contextual Extraction**: Passes the top-ranked context to Gemini with specific field instructions to extract structured data.
4.  **Consolidation**: Merges all extracted fields into a final, nested JSON structure.
5.  **Audit Trail**: Generates a `citations` object mapping each value back to its source chunk index.

---

## Getting Started

### 1. Environment Setup
Create a `.env` file in the root directory with your credentials:
```bash
GOOGLE_APPLICATION_CREDENTIALS="path/to/your/service-account.json"
# Optional: Langfuse for tracing
LANGFUSE_PUBLIC_KEY="pk-lf-..."
LANGFUSE_SECRET_KEY="sk-lf-..."
LANGFUSE_HOST="https://cloud.langfuse.com"
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Pipeline
Place your target `.txt` file in the `input/` directory, update the `input_file` path in `main.py` (line 16), and run:
```bash
python main.py
```
The result will be saved to `output/result.json`.

---

## Tinkering & Iteration

Most of the "intelligence" of the pipeline is governed by `config.py`. This is where you should spend most of your time iterating.

### 1. Improving Extraction (Prompts)
If a field is missing data or extracting incorrectly, edit the `instruction` for that field in `config.py`:
- **Where**: `FIELDS` list in `config.py`.
- **Tip**: Be explicit about what to include and, more importantly, what **not** to include (redirecting to other fields).

### 2. Improving Recall (Retrieval)
If the model says "information not found" but it exists in the document, improve the retrieval queries:
- **Where**: `field["queries"]` in `config.py`.
- **Tip**: Mix exact German procurement terminology (e.g., *Vergabestelle*) with descriptive English/German paraphrases.

### 3. Global Logic & Disambiguation
To change how the model decides between similar fields (e.g., distinguishing between company-level and personnel-level references):
- **Where**: `ROUTING_RULES` in `config.py`.
- **Tip**: This preamble is prepended to every extraction prompt to ensure cross-field consistency.

### 4. Technical Tuning
Adjust the retrieval "funnel" width to balance cost/latency vs. accuracy:
- **`TOP_K_DENSE` / `TOP_K_SPARSE`**: Number of chunks retrieved per query.
- **`TOP_K_RERANK`**: Number of chunks passed to the final extraction step.
- **`CHUNK_SIZE`**: The granularity of the document segments.

---

## Evaluation
Run `python evaluate.py` to validate the extraction results against a ground-truth dataset if available.

---

## Design Philosophy & Blueprint

For a deep dive into the root-cause analysis of extraction failures and the five-stage agentic loop architecture used in this version, see [instructions.md](file:///Users/ervinshaqiri/Documents/Projects/Bidpanion%20v3/instructions.md).

Key architectural layers:
1. **Pre-Processing**: Table reconstruction and LLM metadata enrichment.
2. **Hybrid Index**: Combined BM25 and Vector search.
3. **Multi-Query Retrieval**: LLM-driven query expansion.
4. **Extraction Engine**: Reranked context with structured output.
5. **Agentic Audit**: Null classification and recovery logic.
