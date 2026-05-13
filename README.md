# Bidpanion v5 — Fault-Tolerant Tender Extraction

A production-grade RAG pipeline for German procurement documents, powered by **Temporal** (orchestration), **Vertex AI** (extraction), and **React** (verification).

---

## ⚡️ Quick Start

### 1. Configure Environment
Create a `.env` file and place your GCP service account JSON in the root.
```bash
GOOGLE_APPLICATION_CREDENTIALS="your-key.json"
LANGFUSE_PUBLIC_KEY="pk-lf-..."
LANGFUSE_SECRET_KEY="sk-lf-..."
LANGFUSE_BASE_URL="http://localhost:3000"
```

### 2. Launch the Stack
Start infrastructure, temporal worker, and interactive viewer:
```bash
docker compose up -d --build
```

### 3. Run Analysis
Trigger extraction for a specific document (e.g., `tender_a.txt`):
```bash
docker compose exec bidpanion-worker python -m temporal.run_workflow <filename.txt>
```

### 4. Verify Results
Open the interactive dashboard to verify citations:
👉 **[http://localhost:5173](http://localhost:5173)**


---

## 🛠 Management Commands

| Action | Command |
| :--- | :--- |
| **Stop All** | `docker compose down` |
| **View Logs** | `docker compose logs -f bidpanion-worker` |
| **Temporal UI** | [http://localhost:8080](http://localhost:8080) |
| **Langfuse** | [http://localhost:3000](http://localhost:3000) |
| **Regenerate indices** | `docker compose exec bidpanion-worker python -m temporal.run_workflow --force` |

---

## 🏗 Architecture
- **Worker**: Handles CPU-intensive indexing and LLM extraction (Temporal).
- **Data API**: Serves tender text and `result.json` to the frontend.
- **Viewer**: React dashboard with page-aware citation navigation (5-page overlapping chunks).
