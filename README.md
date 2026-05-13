# Bidpanion v5 — Fault-Tolerant Tender Extraction

Production-grade RAG pipeline powered by **Temporal**, **Vertex AI**, and **React**.

---

## 🚀 Quick Start

### 1. Setup Environment
Create `.env` and place your GCP service account JSON in the root.
```bash
GOOGLE_APPLICATION_CREDENTIALS="key.json"
LANGFUSE_PUBLIC_KEY="pk-lf-..."
LANGFUSE_SECRET_KEY="sk-lf-..."
```

### 2. Launch Stack
Start infrastructure and interactive viewer:
```bash
docker compose up -d --build
```

### 3. Run Extraction
Trigger analysis for a specific document in `input/`:
```bash
docker compose exec bidpanion-worker python -m temporal.run_workflow <tender.txt>
```

### 4. Verify Results
Open the verification dashboard:
👉 **[http://localhost:5173](http://localhost:5173)**

---

## 🛠 Management

| Action | Command |
| :--- | :--- |
| **Stop All** | `docker compose down` |
| **View Worker Logs** | `docker compose logs -f bidpanion-worker` |
| **Temporal UI** | [http://localhost:8080](http://localhost:8080) |
| **Langfuse** | [http://localhost:3000](http://localhost:3000) |
| **Force Re-index** | Add `--force` to the run command |

---

## 🏗 Architecture
- **Worker**: CPU-intensive indexing & LLM extraction (Temporal).
- **Data API**: Serves tender text and dynamic `.json` results.
- **Viewer**: React dashboard with global page-index citation mapping.
