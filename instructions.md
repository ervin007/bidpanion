# Tender RAG — Root-Cause Analysis & Rearchitecture Blueprint

> **Stack:** Python · LangChain · ChromaDB · Gemini 2.5 Flash · BM25 Hybrid Search  
> **Input:** Pre-parsed `.txt` tender files (German language)  
> **Output:** Populated JSON schema with source-cited field values  
> **Goal:** Eliminate null values through hybrid retrieval, multi-query expansion, and agentic audit

---

## Table of Contents

1. [Root-Cause Diagnosis](#1-root-cause-diagnosis)
2. [Redesigned Architecture Overview](#2-redesigned-architecture-overview)
3. [Layer 1 — Pre-Processing](#3-layer-1--pre-processing)
4. [Layer 2 — Hybrid Index (Dense + BM25)](#4-layer-2--hybrid-index-dense--bm25)
5. [Layer 3 — Multi-Query Retrieval](#5-layer-3--multi-query-retrieval)
6. [Layer 4 — Extraction Engine](#6-layer-4--extraction-engine)
7. [Layer 5 — Agentic Audit Loop](#7-layer-5--agentic-audit-loop)
8. [Critical: Document Checklist Extraction](#8-critical-document-checklist-extraction)
9. [Revised config.py](#9-revised-configpy)
10. [Testing & Evaluation Framework](#10-testing--evaluation-framework)
11. [Implementation Roadmap](#11-implementation-roadmap)
12. [New File Structure](#12-new-file-structure)

---

## 1. Root-Cause Diagnosis

Before redesigning anything, understand exactly why fields return null. There are **six distinct failure modes**, each compounding the others.

---

### 1.1 The Single-Query Single-Pass Problem

The original design issues **one query per field** and retrieves chunks once. For a German procurement document this is catastrophically under-specified.

Consider `required_certifications`: the tender may mention it as `Zertifizierungen`, `Nachweise`, `Qualifikationsanforderungen`, `Eignungskriterien`, or buried inside a table labelled `Anlage B`. A single vector query for any one of these synonyms will miss the others entirely.

**Observed in your output:**
- `Standards & Certifications` populated correctly — chunk 644 happened to contain all of them in one place.
- `Personnel Profiles` → null — profiles are spread across multiple sections with different German terminology per lot. No single query spans all three lots.
- `Contract Volume` → null — the information exists but is expressed as a **calculation** (`3 lots × ~15 staff × 1600h × hourly_rate`), not a direct value. No query surfaces a calculation.

---

### 1.2 Dense-Only Retrieval Misses Exact-Match Fields

Dense vector search excels at semantic similarity but performs poorly on **exact-match fields**: `Vergabenummer` (AS240020), specific dates (`16.08.2024 17:00 Uhr`), NUTS codes (`DE929`), EUR amounts, URL strings.

These are high-entropy tokens that embedding models frequently misrepresent in vector space. **BM25 / TF-IDF sparse retrieval handles these with near-perfect recall.** Your system uses only dense retrieval — meaning any field whose value is a code, number, date, or URL is at systematic risk of being null.

| Field Type | Why Dense Search Fails It |
|---|---|
| Vergabenummer | Exact string match — dense may not rank the chunk highly unless query is identical |
| Submission Deadline | Date `16.08.2024 17:00` — dense ranks thematically, BM25 ranks exactly |
| Portal URLs | `bieterzugang.deutsche-evergabe.de` — purely lexical, embeddings add noise |
| EUR Values | Estimated value hidden as calculation — needs keyword + table extraction |
| NUTS Code | `DE929` — rare token, likely low-weight in embedding space |

---

### 1.3 The "Lost in the Middle" Problem Is Insufficiently Solved

The original plan acknowledges "Lost in the Middle" bias and proposes shuffling. **Shuffling 100 chunks at random does not solve the structural problem — it randomises it.**

The real fix is **positional reranking**: send the top-K chunks to a cross-encoder to select the 25 highest-signal chunks before sending to the extraction LLM. With 100 chunks × 1200 tokens = 120k tokens, you are filling most of Gemini 2.5 Flash's window and attention still degrades in the middle 60k tokens.

---

### 1.4 Metadata Is Generated Once and Never Corrected

The chunker detects section headings via regex. German tender documents are not uniform: some use `Abschnitt`, others use `Los`, others use Roman numerals, and PDF-artifact whitespace breaks the regex entirely.

Once a chunk gets wrong metadata (`section='Unbekannter Abschnitt'`), it is **permanently wrong**. There is no LLM-based metadata correction pass. Retrieval queries that rely on section context will consistently fail for mis-labelled chunks.

---

### 1.5 The Auditor Has No Ground Truth to Check Against

The Delta-Auditor consolidates `---` delimited multi-pass results, but it has **no mechanism to detect when all three passes returned null** (retrieval failed) versus when all three returned conflicting values (extraction ambiguous). These two failure modes require completely different recovery strategies. The current design treats them identically.

---

### 1.6 No Table-Aware Parsing

Tender documents — especially German public procurement — store critical data in **structured tables**: award criteria weights, lot volumes, deadline matrices, required document checklists.

When a PDF-to-TXT converter flattens these tables, the spatial relationships are destroyed. `60%` and `40%` may appear on adjacent lines with no indication they refer to `Leistungspunkte` and `Preispunkte` respectively. Your chunker has no table reconstruction step, so table-derived fields (contract volume, award weights, document checklist) are **systematically under-extracted**.

---

## 2. Redesigned Architecture Overview

The rearchitected pipeline replaces single-pass retrieval with a **five-stage agentic loop**. Each stage has a defined responsibility and guards against a specific failure mode.

```
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 1: PRE-PROCESSING                                        │
│  Table reconstruction → LLM metadata enrichment → anchoring    │
├─────────────────────────────────────────────────────────────────┤
│  STAGE 2: HYBRID INDEX                                          │
│  ChromaDB dense (text-embedding-004) + BM25 sparse (rank_bm25) │
├─────────────────────────────────────────────────────────────────┤
│  STAGE 3: MULTI-QUERY RETRIEVAL                                 │
│  LLM query expansion (4 German variants) + RRF fusion           │
├─────────────────────────────────────────────────────────────────┤
│  STAGE 4: EXTRACTION                                            │
│  Grouped fields + cross-encoder reranking + structured output   │
├─────────────────────────────────────────────────────────────────┤
│  STAGE 5: AGENTIC AUDIT                                         │
│  Null classification → targeted re-retrieval → consistency check│
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Layer 1 — Pre-Processing

### 3.1 Table Reconstruction Before Chunking

**This is the single most impactful change you can make.** Before passing the raw `.txt` to the chunker, run a table reconstruction pass that converts linearised table text back into structured key-value pairs.

```python
# ingestion/table_reconstructor.py

import re
from langchain.schema import Document


def reconstruct_tables(raw_text: str) -> str:
    """
    Detect linearised tables in plain text and reformat them as
    explicit key:value pairs so the chunker preserves associations.
    """
    lines = raw_text.split("\n")
    output_lines = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # Detect table-like lines: multiple tab/pipe separators or
        # consistent multi-column spacing (≥2 separators per line)
        if _is_table_row(line):
            table_lines = []
            while i < len(lines) and (_is_table_row(lines[i]) or lines[i].strip() == ""):
                if lines[i].strip():
                    table_lines.append(lines[i])
                i += 1

            if len(table_lines) >= 2:
                reconstructed = _format_table(table_lines)
                output_lines.append("TABLE_START")
                output_lines.extend(reconstructed)
                output_lines.append("TABLE_END")
                continue
        else:
            output_lines.append(line)

        i += 1

    return "\n".join(output_lines)


def _is_table_row(line: str) -> bool:
    # Tabs, pipes, or ≥3 consecutive spaces as column separators
    return (
        line.count("\t") >= 2
        or line.count("|") >= 2
        or len(re.findall(r"   +", line)) >= 2
    )


def _format_table(lines: list[str]) -> list[str]:
    """Convert detected table rows to Header: Value pairs."""
    # Split each row on tab/pipe/multi-space
    rows = [re.split(r"\t|\|{1,2}|   +", l.strip()) for l in lines]
    rows = [[cell.strip() for cell in row if cell.strip()] for row in rows]

    if not rows:
        return lines

    headers = rows[0]
    result = []
    for data_row in rows[1:]:
        for j, cell in enumerate(data_row):
            if j < len(headers) and cell:
                result.append(f"{headers[j]}: {cell}")
    return result if result else [" | ".join(c for r in rows for c in r)]
```

**Critical rule:** Award criteria tables (containing weights like 60%/40%) must **never be split across chunks**. Force the entire table into one chunk:

```python
# In chunker.py — after table reconstruction, treat TABLE_START...TABLE_END as atomic units
def load_and_chunk(txt_path: str) -> list[Document]:
    with open(txt_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    # Reconstruct tables first
    raw_text = reconstruct_tables(raw_text)

    # Extract atomic blocks (tables) that must not be split
    atomic_blocks = re.findall(r"TABLE_START.*?TABLE_END", raw_text, re.DOTALL)
    
    # Replace with placeholders, chunk the rest, reinsert as atomic chunks
    # ... (see full implementation below)
```

---

### 3.2 LLM Metadata Enrichment Pass

After initial chunking, run a **one-time LLM pass** over every chunk to generate richer metadata. This is paid at index time, not retrieval time.

```python
# ingestion/metadata_enricher.py

import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import Document, SystemMessage, HumanMessage
from config import GOOGLE_API_KEY

ENRICHMENT_SYSTEM = """Du bist ein Klassifikator für deutsche Vergabeunterlagen.
Antworte NUR mit einem gültigen JSON-Objekt, kein weiterer Text."""

ENRICHMENT_PROMPT = """Klassifiziere diesen Textausschnitt aus einem deutschen Vergabedokument.

Chunk:
{chunk_text}

Antworte mit exakt diesem JSON:
{{
  "content_type": "<table|list|prose|header|deadline|legal|contact>",
  "lot_scope": "<[1]|[2]|[3]|[1,2]|[1,3]|[2,3]|[1,2,3]|[]>",
  "contains_deadline": <true|false>,
  "contains_value": <true|false>,
  "contains_document_list": <true|false>,
  "primary_entity": "<authority|candidate|document|personnel|procedure|award>",
  "corrected_section": "<best German section title for this chunk>",
  "german_keywords": ["<keyword1>", "<keyword2>", "<keyword3>"]
}}"""


def enrich_metadata_batch(documents: list[Document], batch_size: int = 10) -> list[Document]:
    """
    Run LLM metadata enrichment on all chunks.
    Batches calls to reduce latency. Called ONCE at index time.
    """
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.0,
    )
    enriched = []

    for i in range(0, len(documents), batch_size):
        batch = documents[i : i + batch_size]
        for doc in batch:
            try:
                prompt = ENRICHMENT_PROMPT.format(chunk_text=doc.page_content[:800])
                response = llm.invoke([
                    SystemMessage(content=ENRICHMENT_SYSTEM),
                    HumanMessage(content=prompt),
                ])
                meta = json.loads(response.content)
                doc.metadata.update(meta)
            except Exception as e:
                # Fallback: keep existing regex-detected metadata
                doc.metadata["enrichment_error"] = str(e)
            enriched.append(doc)

    return enriched
```

**Metadata fields added per chunk:**

| Field | Type | Purpose |
|---|---|---|
| `content_type` | enum | Filter deadline chunks, list chunks, table chunks separately |
| `lot_scope` | list[int] | Filter by lot for per-lot field extraction |
| `contains_deadline` | bool | Deadline-specific retrieval filter |
| `contains_value` | bool | Numeric extraction mode trigger |
| `contains_document_list` | bool | Document checklist extractor filter |
| `primary_entity` | enum | Scope retrieval to relevant document section |
| `corrected_section` | str | Overrides broken regex-detected section heading |
| `german_keywords` | list[str] | BM25 boosting tokens |

---

### 3.3 Structural Anchoring for Mid-List Chunks

```python
# In chunker.py — inject parent heading into chunks that start mid-list

def inject_context_prefix(chunk_text: str, section_heading: str) -> str:
    """
    If a chunk starts with a list item or continuation, prepend
    the section heading so retrieval queries see the context.
    """
    list_start_patterns = [
        r"^\s*[-•]\s",           # bullet
        r"^\s*\d+[\.\)]\s",      # numbered list
        r"^\s*[a-z]\)\s",        # lettered list
        r"^\s*(und|sowie|oder)\s", # continuation conjunction
    ]
    for pat in list_start_patterns:
        if re.match(pat, chunk_text, re.IGNORECASE):
            return f"[Kontext: {section_heading}]\n{chunk_text}"
    return chunk_text
```

---

## 4. Layer 2 — Hybrid Index (Dense + BM25)

### 4.1 Why BM25 Is Non-Negotiable for Tender Documents

| Retrieval Mode | What It Catches |
|---|---|
| Dense (ChromaDB) | Semantically related chunks even when terminology varies |
| BM25 (rank-bm25) | Exact match: codes, dates, URLs, form names, regulation refs |
| RRF Fusion | Merges ranked lists without requiring score normalisation |

**Fields that will remain null without BM25:**
- `Vergabenummer` — exact alphanumeric code
- All deadline fields — exact date+time strings
- Portal URLs — lexical tokens, not semantic
- Form references (E1–E6, Mantelbogen) — rare exact strings
- NUTS codes, CPV codes — structured codes

---

### 4.2 BM25 Implementation

```bash
pip install rank-bm25
```

```python
# vectorstore/bm25_index.py

import re
import pickle
from rank_bm25 import BM25Okapi
from langchain.schema import Document


def tokenize_german(text: str) -> list[str]:
    """Lowercase tokenisation preserving German umlauts."""
    text = text.lower()
    return re.findall(r"[a-zäöüß0-9]+", text)


def build_bm25(documents: list[Document]) -> tuple[BM25Okapi, list[Document]]:
    """Build BM25 index aligned with document list."""
    tokenized = [tokenize_german(d.page_content) for d in documents]
    bm25 = BM25Okapi(tokenized)
    return bm25, documents


def bm25_search(
    bm25: BM25Okapi,
    docs: list[Document],
    query: str,
    k: int = 30,
) -> list[tuple[Document, float]]:
    """Return top-k (doc, score) pairs for a query."""
    tokens = tokenize_german(query)
    scores = bm25.get_scores(tokens)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [(docs[i], float(scores[i])) for i in top_indices]


def save_bm25(bm25: BM25Okapi, path: str):
    with open(path, "wb") as f:
        pickle.dump(bm25, f)


def load_bm25(path: str) -> BM25Okapi:
    with open(path, "rb") as f:
        return pickle.load(f)
```

---

### 4.3 Reciprocal Rank Fusion

```python
# vectorstore/fusion.py

from langchain.schema import Document


def rrf_merge(
    dense_results: list[Document],
    sparse_results: list[tuple[Document, float]],
    k: int = 60,
    top_n: int = 100,
) -> list[Document]:
    """
    Reciprocal Rank Fusion of dense and sparse ranked lists.
    Does not require score normalisation — works on ranks alone.

    k=60 is the standard RRF constant (Robertson et al. 2009).
    """
    scores: dict[int, float] = {}
    doc_map: dict[int, Document] = {}

    for rank, doc in enumerate(dense_results):
        cid = doc.metadata["chunk_index"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
        doc_map[cid] = doc

    for rank, (doc, _) in enumerate(sparse_results):
        cid = doc.metadata["chunk_index"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
        doc_map[cid] = doc

    sorted_ids = sorted(scores, key=scores.__getitem__, reverse=True)[:top_n]
    return [doc_map[cid] for cid in sorted_ids if cid in doc_map]
```

---

### 4.4 Updated store.py — Hybrid Retrieval

```python
# vectorstore/store.py  (additions to existing file)

from vectorstore.bm25_index import bm25_search
from vectorstore.fusion import rrf_merge


def hybrid_retrieve(
    vectorstore,
    bm25_index,
    bm25_docs: list,
    query: str,
    dense_k: int = 60,
    sparse_k: int = 40,
    final_k: int = 100,
    metadata_filter: dict = None,
) -> list:
    """
    Retrieve using both dense MMR and BM25, fuse with RRF.
    metadata_filter: ChromaDB where clause dict (optional).
    """
    search_kwargs = {"k": dense_k, "fetch_k": dense_k * 3, "lambda_mult": 0.6}
    if metadata_filter:
        search_kwargs["filter"] = metadata_filter

    retriever = vectorstore.as_retriever(
        search_type="mmr", search_kwargs=search_kwargs
    )
    dense_results = retriever.invoke(query)
    sparse_results = bm25_search(bm25_index, bm25_docs, query, k=sparse_k)

    return rrf_merge(dense_results, sparse_results, top_n=final_k)
```

---

## 5. Layer 3 — Multi-Query Retrieval

### 5.1 Query Expansion via LLM

For each field, generate **3–5 distinct German queries** before retrieval. This is the single most effective technique for recall improvement in multilingual RAG.

```python
# extraction/query_expander.py

import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import SystemMessage, HumanMessage
from config import GOOGLE_API_KEY

EXPANDER_SYSTEM = """Du bist ein Experte für deutsches Vergaberecht und öffentliche Ausschreibungen.
Antworte NUR mit einem JSON-Array aus Strings."""

EXPANDER_PROMPT = """Ich suche folgende Information in einem deutschen Vergabedokument:

Feld: {field_name}
Beschreibung: {field_description}

Generiere 4 verschiedene deutsche Suchanfragen, die diese Information finden würden.
Variiere: Terminologie, Formulierung, Kontext (rechtlich vs. praktisch vs. Tabellenüberschrift).

Antworte NUR mit einem JSON-Array mit 4 Strings.
Beispiel: ["Abgabefrist Angebot", "Einreichungsfrist Datum", "Fristablauf Angebotsabgabe", "bis zum Datum Uhrzeit"]"""


def expand_queries(field_name: str, field_description: str) -> list[str]:
    """Generate 4 German search query variants for a field."""
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.2,
    )
    try:
        response = llm.invoke([
            SystemMessage(content=EXPANDER_SYSTEM),
            HumanMessage(content=EXPANDER_PROMPT.format(
                field_name=field_name,
                field_description=field_description,
            )),
        ])
        queries = json.loads(response.content)
        # Always include the original query as a fallback
        return list(dict.fromkeys([field_name] + queries))  # deduplicate, preserve order
    except Exception:
        return [field_name]  # fallback to original
```

---

### 5.2 Field-Type Specific Query Strategies

| Field Category | Specialised Query Strategy |
|---|---|
| Deadline fields | Add: `Datum`, `Uhr`, `Frist`, `bis zum` — filter by `contains_deadline=True` |
| Value/amount fields | Add: `EUR`, `Stunden`, `Netto`, `Betrag` — filter by `contains_value=True` |
| Legal/eligibility | Use §§ references: `§ 123 GWB`, `§ 124`, `Eignungskriterien`, `Ausschlussgründe` |
| Document lists | Use: `Formblatt`, `Anlage`, `Nachweis`, `einzureichen`, `beizufügen`, `Unterlage` |
| Personnel profiles | **Query per lot**: `Los 1 Anforderungen`, `Los 2 Senior`, `Los 3 Assistenz` — 3 queries minimum |
| Contact/authority | Use: `Auftraggeber`, `Vergabestelle`, `Kontakt`, `Ansprechpartner`, `Adresse` |

---

### 5.3 Multi-Query Retrieval with Deduplication

```python
# extraction/multi_query_retriever.py

from vectorstore.store import hybrid_retrieve
from langchain.schema import Document


def multi_query_retrieve(
    vectorstore,
    bm25_index,
    bm25_docs: list,
    queries: list[str],
    metadata_filter: dict = None,
    per_query_k: int = 40,
    final_k: int = 100,
) -> list[Document]:
    """
    Run hybrid retrieval for each query, merge by chunk_index, deduplicate.
    Returns up to final_k unique chunks ranked by combined RRF score.
    """
    all_results: dict[int, Document] = {}
    rrf_scores: dict[int, float] = {}

    for q_rank, query in enumerate(queries):
        results = hybrid_retrieve(
            vectorstore, bm25_index, bm25_docs,
            query=query,
            dense_k=per_query_k,
            sparse_k=per_query_k // 2,
            final_k=per_query_k,
            metadata_filter=metadata_filter,
        )
        for doc_rank, doc in enumerate(results):
            cid = doc.metadata["chunk_index"]
            # RRF across queries: earlier queries get slight priority
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (60 + doc_rank + 1)
            all_results[cid] = doc

    sorted_ids = sorted(rrf_scores, key=rrf_scores.__getitem__, reverse=True)[:final_k]
    return [all_results[cid] for cid in sorted_ids]
```

---

## 6. Layer 4 — Extraction Engine

### 6.1 Field Grouping Strategy (Revised)

Split into **10 semantic groups** — fields that live in different sections of a tender must not be extracted together (LLM attention degrades).

| Group | Fields | Notes |
|---|---|---|
| A: Identity | `contracting_authority_*` (5 fields) | All from section `Auftraggeber` |
| B: Scope | `contract_title`, `contract_description`, `cpv_code`, `place_of_performance` | |
| C: Timeline | All deadline and date fields (6 fields) | Use date-specific query strategy |
| D: Value | `estimated_value`, `currency`, `lot_values`, `contract_type` | Numeric mode |
| E: Procedure | `procedure_type`, `framework_agreement`, `lots_allowed`, `subcontracting_allowed` | |
| F: Eligibility | `required_certifications`, `minimum_turnover`, `required_references`, `exclusion_grounds` | |
| G: Documents | **All form/document checklist fields** | List mode, enumerate every item |
| H: Personnel | **One group PER LOT** — 3 separate extraction calls | Los 1 / Los 2 / Los 3 |
| I: Award | `award_criteria`, `price_weight_percent`, `quality_weight_percent`, `evaluation_method` | |
| J: Submission | `tender_documents_url`, `document_fee`, `tender_platform` | BM25-boosted |

---

### 6.2 Cross-Encoder Reranking

After RRF fusion retrieves 100 candidates, rerank to select the **top 25 highest-signal chunks**.

```bash
pip install sentence-transformers
```

```python
# extraction/reranker.py

from sentence_transformers import CrossEncoder
from langchain.schema import Document

# For German documents consider: 'deutsche-telekom/gbert-large-paraphrase-cosine'
# ms-marco is English but still effective for reranking German chunk relevance
_reranker = None


def get_reranker(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(model_name)
    return _reranker


def rerank(query: str, docs: list[Document], top_k: int = 25) -> list[Document]:
    """
    Rerank retrieved chunks using a cross-encoder.
    Reduces context from 100 chunks to top_k highest-relevance chunks.
    """
    if not docs:
        return docs
    reranker = get_reranker()
    pairs = [(query, d.page_content) for d in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[:top_k]]
```

---

### 6.3 Structured Output via Gemini response_schema

Replace ad-hoc JSON parsing with **Gemini's native structured output**. Eliminates JSON parse errors entirely.

```python
# extraction/extractor.py  (revised)

import json
import google.generativeai as genai
from config import GOOGLE_API_KEY, LLM_MODEL

genai.configure(api_key=GOOGLE_API_KEY)


def extract_group(
    field_group: list[dict],
    context_chunks: list,
    response_schema: dict,
) -> dict:
    """
    Extract a group of related fields from context chunks.
    Uses Gemini's response_schema for guaranteed valid JSON output.
    """
    model = genai.GenerativeModel(
        model_name=LLM_MODEL,
        system_instruction=SYSTEM_PROMPT,
    )

    formatted_context = _format_chunks(context_chunks)
    field_names = [f["id"] for f in field_group]
    instructions = {f["id"]: f["instruction"] for f in field_group}

    user_prompt = f"""Extrahiere die folgenden Felder aus den Kontext-Chunks:

Felder und Anweisungen:
{json.dumps(instructions, ensure_ascii=False, indent=2)}

Kontext-Chunks:
{formatted_context}

Regeln:
- Setze Felder auf null wenn nicht im Kontext vorhanden
- Erfinde keine Informationen
- Füge für jedes nicht-null Feld ein {{feldname}}__chunk Feld mit dem chunk_index hinzu
- Für Listen: gib ALLE gefundenen Items zurück, nicht nur die ersten
"""

    response = model.generate_content(
        user_prompt,
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json",
            response_schema=response_schema,
            temperature=0.0,
        ),
    )

    return json.loads(response.text)


def _format_chunks(chunks: list) -> str:
    parts = []
    for doc in chunks:
        idx = doc.metadata.get("chunk_index", "?")
        section = doc.metadata.get("corrected_section") or doc.metadata.get("section", "?")
        content_type = doc.metadata.get("content_type", "?")
        parts.append(
            f"[Chunk {idx} | {section} | type:{content_type}]\n{doc.page_content}"
        )
    return "\n---\n".join(parts)
```

---

### 6.4 Extraction Modes

Define `extraction_mode` per field in config. Each mode uses a different prompt strategy.

```python
# extraction/prompts.py

EXTRACTION_MODE_HINTS = {
    "exact": (
        "Extrahiere einen einzigen, präzisen Wert. "
        "Keine Interpretation — nur das was explizit im Text steht."
    ),
    "list": (
        "Extrahiere ALLE zutreffenden Items als Array. "
        "Vollständigkeit ist wichtiger als Präzision. "
        "Lieber zu viele als zu wenige Items."
    ),
    "narrative": (
        "Fasse den relevanten Inhalt in einem zusammenhängenden deutschen Text zusammen. "
        "Maximal 300 Wörter. Keine Bullet Points."
    ),
    "numeric": (
        "Extrahiere numerische Werte. "
        "Bei Berechnungen: führe die Berechnung aus und gib das Ergebnis zurück. "
        "Beispiel: '3 Mitarbeiter × 1600 Stunden = 4800' → gib 4800 zurück."
    ),
}
```

---

## 7. Layer 5 — Agentic Audit Loop

### 7.1 Null Classification — Three Types

| Null Type | Meaning | Recovery Action |
|---|---|---|
| TYPE 1: Not in document | Field genuinely absent from this tender | Accept null, mark as `confirmed_absent` |
| TYPE 2: Retrieval miss | Field exists but retrieval failed to surface the chunk | Re-retrieve with alternative queries + BM25 boost |
| TYPE 3: Extraction miss | Chunk was retrieved but LLM failed to extract | Re-run with explicit table-parsing / list-mode prompt |

---

### 7.2 Null Classifier

```python
# audit/null_classifier.py

import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import SystemMessage, HumanMessage
from config import GOOGLE_API_KEY


def classify_null(
    field_name: str,
    field_description: str,
    tender_context: dict,  # other already-extracted fields for context
    llm=None,
) -> str:
    """
    Ask the LLM whether a null field is expected or a retrieval gap.
    Returns: 'confirmed_absent' | 'retrieval_miss' | 'extraction_miss'
    """
    if llm is None:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=GOOGLE_API_KEY,
            temperature=0.0,
        )

    context_summary = {
        k: v for k, v in tender_context.items()
        if v is not None and not k.endswith("__chunk")
    }

    prompt = f"""Ein Feld wurde in einem deutschen Vergabedokument nicht gefunden.

Feld: {field_name}
Beschreibung: {field_description}

Bereits extrahierte Felder (Kontext):
{json.dumps(context_summary, ensure_ascii=False, indent=2)}

Frage: Ist es wahrscheinlich, dass dieses Feld:
a) 'confirmed_absent' — in dieser Art von Ausschreibung typischerweise nicht vorkommt
b) 'retrieval_miss' — vorhanden, aber die Suche hat es nicht gefunden
c) 'extraction_miss' — die relevanten Chunks wurden gefunden, aber der Wert nicht extrahiert

Antworte NUR mit einem dieser drei Strings: confirmed_absent | retrieval_miss | extraction_miss"""

    response = llm.invoke([
        SystemMessage(content="Du bist ein Experte für deutsche Vergabeverfahren."),
        HumanMessage(content=prompt),
    ])
    result = response.content.strip().lower()
    if result in ("confirmed_absent", "retrieval_miss", "extraction_miss"):
        return result
    return "retrieval_miss"  # default to retry
```

---

### 7.3 Recovery Agent

```python
# audit/recovery_agent.py

import logging
from extraction.query_expander import expand_queries
from extraction.multi_query_retriever import multi_query_retrieve
from extraction.extractor import extract_group
from extraction.reranker import rerank
from audit.null_classifier import classify_null

logger = logging.getLogger(__name__)


def run_recovery_pass(
    extracted: dict,
    field_configs: list[dict],
    vectorstore,
    bm25_index,
    bm25_docs: list,
    llm=None,
) -> dict:
    """
    Identify all null fields, classify them, and attempt targeted recovery.
    Returns updated extracted dict with recovered values and audit annotations.
    """
    null_fields = [
        cfg for cfg in field_configs
        if extracted.get(cfg["id"]) is None
    ]

    logger.info(f"Recovery pass: {len(null_fields)} null fields to process")

    for field_cfg in null_fields:
        fid = field_cfg["id"]
        logger.info(f"  Processing null: {fid}")

        # Step 1: Classify the null
        null_type = classify_null(
            field_name=fid,
            field_description=field_cfg.get("instruction", ""),
            tender_context=extracted,
            llm=llm,
        )

        extracted[f"{fid}__null_type"] = null_type
        logger.info(f"    Null type: {null_type}")

        if null_type == "confirmed_absent":
            continue  # Expected — do not retry

        # Step 2: Targeted re-retrieval with expanded queries
        alt_queries = expand_queries(fid, field_cfg.get("instruction", ""))

        # Boost BM25 by adding field-specific German keywords
        keyword_queries = field_cfg.get("queries", []) + alt_queries

        chunks = multi_query_retrieve(
            vectorstore, bm25_index, bm25_docs,
            queries=keyword_queries,
            metadata_filter=field_cfg.get("metadata_filter"),
            per_query_k=30,
            final_k=60,
        )

        # Step 3: Rerank to top 20 for focused extraction
        primary_query = field_cfg.get("queries", [fid])[0]
        reranked_chunks = rerank(primary_query, chunks, top_k=20)

        if not reranked_chunks:
            extracted[f"{fid}__null_type"] = "confirmed_absent"
            continue

        # Step 4: Re-extract with table/list mode forced
        forced_cfg = {**field_cfg, "extraction_mode": "list" if null_type == "extraction_miss" else field_cfg.get("extraction_mode", "exact")}

        result = extract_group(
            field_group=[forced_cfg],
            context_chunks=reranked_chunks,
            response_schema={fid: {"type": "string"}},
        )

        recovered_value = result.get(fid)
        if recovered_value is not None:
            extracted[fid] = recovered_value
            extracted[f"{fid}__recovered"] = True
            logger.info(f"    ✓ Recovered: {str(recovered_value)[:80]}")
        else:
            logger.info(f"    ✗ Recovery failed — marking as confirmed_absent")
            extracted[f"{fid}__null_type"] = "confirmed_absent"

    return extracted
```

---

### 7.4 Consistency Checks

```python
# audit/consistency_checks.py

from datetime import datetime
import re
import logging

logger = logging.getLogger(__name__)


def run_consistency_checks(data: dict) -> list[str]:
    """
    Post-extraction logical validation.
    Returns list of warning strings (empty = all checks passed).
    """
    warnings = []

    # 1. Award criteria weights must sum to ~100%
    price = data.get("price_weight_percent")
    quality = data.get("quality_weight_percent")
    if price is not None and quality is not None:
        total = price + quality
        if not (95 <= total <= 105):
            warnings.append(
                f"Award weights sum to {total}%, expected ~100% "
                f"(price={price}%, quality={quality}%)"
            )

    # 2. Deadline ordering
    sub_deadline = _parse_date_de(data.get("submission_deadline", ""))
    q_deadline   = _parse_date_de(data.get("question_deadline", ""))
    if sub_deadline and q_deadline and q_deadline >= sub_deadline:
        warnings.append(
            f"question_deadline ({q_deadline}) must be before "
            f"submission_deadline ({sub_deadline})"
        )

    # 3. CPV code format
    cpv = data.get("cpv_code", "")
    if cpv and not re.match(r"^\d{8}-\d$", cpv):
        warnings.append(f"CPV code format invalid: '{cpv}' (expected: 72000000-5)")

    # 4. Currency consistency
    fields_with_values = [
        data.get("estimated_value"),
        data.get("minimum_turnover"),
    ]
    if any(v is not None for v in fields_with_values):
        if not data.get("currency"):
            warnings.append("Monetary fields present but currency is null")

    # 5. Lot headcount sanity
    for lot_n in [1, 2, 3]:
        headcount_key = f"lot_{lot_n}_headcount"
        if data.get(headcount_key) is not None:
            expected_hours = data[headcount_key] * 1600
            actual_hours = data.get(f"lot_{lot_n}_hours_per_year")
            if actual_hours and abs(actual_hours - expected_hours) > 800:
                warnings.append(
                    f"Los {lot_n}: headcount × 1600 = {expected_hours}h "
                    f"but lot_hours = {actual_hours}h"
                )

    # 6. Estimated value plausibility
    val = data.get("estimated_value")
    if val is not None and val <= 0:
        warnings.append(f"estimated_value is non-positive: {val}")

    return warnings


def _parse_date_de(date_str: str):
    if not date_str:
        return None
    for fmt in ("%d.%m.%Y", "%d.%m.%Y %H:%M", "%d.%m.%Y %H:%M:%S"):
        try:
            return datetime.strptime(date_str.strip()[:19], fmt)
        except ValueError:
            continue
    return None
```

---

## 8. Critical: Document Checklist Extraction

> **Highest-stakes field.** A missed required document means a disqualified bid. This warrants a dedicated pipeline separate from the general extraction flow.

### 8.1 Why Document Lists Fail

- Spread across multiple sections: `Teilnahmebedingungen`, `Eignungsnachweise`, `Angebotsunterlagen`, `Formblätter`
- Inconsistent verbs: `einzureichen`, `beizufügen`, `vorzulegen`, `nachzuweisen`, `anzugeben`
- Mix of mandatory (`muss`, `ist`, `sind`) and optional (`kann`, `soweit vorhanden`)
- Form numbers (E1–E6, Mantelbogen) appear without descriptive context nearby
- Some documents are implicit (complete bid = price sheet included)

### 8.2 The Eight Retrieval Queries

Always run **all eight** — never reduce this list:

```python
DOCUMENT_QUERIES = [
    "einzureichende Unterlagen Angebot Nachweise",
    "Formblätter Anlagen beizufügen",
    "Eignungsnachweis vorzulegen einreichen",
    "Teilnahmebedingungen Unterlagen Pflicht",
    "Angebotsbestandteile Checkliste vollständig",
    "Formblatt E1 E2 E3 E4 E5 E6 Mantelbogen",
    "Pflichtangaben Bieter Eigenerklärung",
    "Handelsregister Betriebshaftpflicht Versicherung Nachweis",
]
```

### 8.3 Dedicated Document Extractor

```python
# extraction/document_extractor.py

import json
import logging
from extraction.multi_query_retriever import multi_query_retrieve
from extraction.reranker import rerank
from config import GOOGLE_API_KEY, LLM_MODEL
import google.generativeai as genai

logger = logging.getLogger(__name__)

DOCUMENT_LIST_SYSTEM = """Du bist ein Experte für deutsche Vergabeunterlagen.
Deine Aufgabe ist die vollständige Erfassung ALLER einzureichenden Dokumente.
Antworte NUR mit einem gültigen JSON-Objekt."""

DOCUMENT_LIST_PROMPT = """Erfasse ALLE Dokumente, Nachweise, Formulare und Erklärungen,
die Bieter mit ihrem Angebot einreichen müssen.

Kontext-Chunks:
{context}

Gib für jedes Dokument zurück:
- name: Dokumentenname/-bezeichnung
- mandatory: true/false (Pflicht vs. optional)
- form_reference: Formblatt-Nummer falls vorhanden (z.B. "E1", "Mantelbogen")
- lot_scope: welche Lose betroffen [1,2,3] oder "all"
- section_reference: Abschnitt im Dokument wo gefordert

WICHTIG: Vollständigkeit ist kritisch. Lieber zu viele als zu wenige Items.
Schließe ein: Eigenerklärungen, Formblätter, Nachweise, Referenzen, Lebensläufe,
Versicherungsnachweise, Handelsregisterauszüge, EEE, und alle sonstigen Anlagen."""


def extract_document_checklist(
    vectorstore,
    bm25_index,
    bm25_docs: list,
    n_passes: int = 3,
) -> list[dict]:
    """
    Run the dedicated document checklist extraction pipeline.
    Uses 3 passes with different query subsets, then deduplicates.
    """
    all_documents: list[dict] = []

    for pass_num in range(n_passes):
        logger.info(f"  Document extraction pass {pass_num + 1}/{n_passes}")

        # Rotate query subsets across passes for coverage
        query_subset = DOCUMENT_QUERIES[pass_num::n_passes] + DOCUMENT_QUERIES[:pass_num]

        chunks = multi_query_retrieve(
            vectorstore, bm25_index, bm25_docs,
            queries=query_subset,
            metadata_filter={"contains_document_list": True},
            per_query_k=20,
            final_k=60,
        )

        # Also retrieve ALL list-type chunks regardless of query
        list_chunks = multi_query_retrieve(
            vectorstore, bm25_index, bm25_docs,
            queries=["Formblatt Anlage einzureichen"],
            metadata_filter={"content_type": "list"},
            final_k=30,
        )

        # Merge, deduplicate by chunk_index
        combined = {d.metadata["chunk_index"]: d for d in chunks + list_chunks}
        context_chunks = list(combined.values())
        context_text = "\n---\n".join(
            f"[Chunk {d.metadata['chunk_index']}]\n{d.page_content}"
            for d in context_chunks
        )

        model = genai.GenerativeModel(model_name=LLM_MODEL, system_instruction=DOCUMENT_LIST_SYSTEM)
        response = model.generate_content(
            DOCUMENT_LIST_PROMPT.format(context=context_text),
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                temperature=0.0,
            ),
        )

        try:
            result = json.loads(response.text)
            docs = result if isinstance(result, list) else result.get("documents", [])
            all_documents.extend(docs)
        except Exception as e:
            logger.warning(f"  Pass {pass_num + 1} parse error: {e}")

    # Deduplicate by normalised name
    return _deduplicate_documents(all_documents)


def _deduplicate_documents(docs: list[dict]) -> list[dict]:
    seen = {}
    for doc in docs:
        name = doc.get("name", "").lower().strip()
        name = name.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue").replace("ß", "ss")
        if name not in seen:
            seen[name] = doc
        else:
            # Merge: prefer mandatory=True, merge lot_scope
            existing = seen[name]
            if doc.get("mandatory"):
                existing["mandatory"] = True
            existing_lots = set(existing.get("lot_scope", []) if isinstance(existing.get("lot_scope"), list) else [])
            new_lots = set(doc.get("lot_scope", []) if isinstance(doc.get("lot_scope"), list) else [])
            if existing_lots or new_lots:
                existing["lot_scope"] = sorted(existing_lots | new_lots)
    return list(seen.values())
```

### 8.4 Standard Forms Cross-Check

After extraction, validate against known standard German procurement forms:

```python
# audit/forms_validator.py

STANDARD_FORMS = {
    "EEE": {
        "full_name": "Einheitliche Europäische Eigenerklärung",
        "condition": "always",  # mandatory in all EU-threshold tenders
    },
    "Mantelbogen": {
        "full_name": "Mantelbogen / Angebotsvordruck",
        "condition": "always",
    },
    "§ 123 GWB": {
        "full_name": "Eigenerklärung Ausschlussgründe § 123 GWB",
        "condition": "always",
    },
    "§ 124 GWB": {
        "full_name": "Eigenerklärung Ausschlussgründe § 124 GWB",
        "condition": "always",
    },
    "Handelsregisterauszug": {
        "full_name": "Handelsregisterauszug oder Eigenerklärung",
        "condition": "always",
    },
    "Betriebshaftpflicht": {
        "full_name": "Betriebshaftpflichtversicherung Nachweis",
        "condition": "it_services",  # likely for IT tenders
    },
}


def validate_document_completeness(
    extracted_docs: list[dict],
    procedure_type: str = "",
) -> list[str]:
    """Check extracted document list against known standard forms."""
    warnings = []
    extracted_names_lower = " ".join(
        d.get("name", "").lower() for d in extracted_docs
    )

    for form_key, form_info in STANDARD_FORMS.items():
        condition = form_info["condition"]
        if condition == "always" or (condition == "it_services" and "it" in procedure_type.lower()):
            if form_key.lower() not in extracted_names_lower:
                warnings.append(
                    f"Standard form possibly missing: '{form_info['full_name']}' ({form_key})"
                )

    return warnings
```

---

## 9. Revised config.py

```python
# config.py  (complete revised version)

import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY       = os.getenv("GOOGLE_API_KEY")
EMBEDDING_MODEL      = "text-embedding-004"
LLM_MODEL            = "gemini-2.5-flash"
CHROMA_PERSIST_DIR   = "./chroma_db"
CHROMA_COLLECTION    = "tenders"
BM25_INDEX_PATH      = "./bm25_index.pkl"

# Chunking
CHUNK_SIZE    = 1200
CHUNK_OVERLAP = 200
SEPARATORS    = ["\n\n", "\n", ".", " "]

# Retrieval
TOP_K_DENSE   = 60
TOP_K_SPARSE  = 40
TOP_K_FINAL   = 100
TOP_K_RERANK  = 25   # chunks sent to extraction LLM after reranking

# Field definitions — extended schema
# extraction_mode: exact | list | narrative | numeric
# mandatory: triggers recovery pass if null
# queries: 4 German queries (do not reduce below 3)
# metadata_filter: ChromaDB where clause for filtered retrieval

FIELDS = [
    # ── Group A: Contracting Authority ──────────────────────────────────────
    {
        "id": "contracting_authority_name",
        "group": "A",
        "path": ["Contracting Authority", "name"],
        "queries": [
            "Auftraggeber Name Vergabestelle",
            "Öffentlicher Auftraggeber Bezeichnung",
            "Vergabenummer Auftraggeber",
            "Beschaffungsstelle Name Adresse",
        ],
        "instruction": "Full legal name of the contracting authority.",
        "extraction_mode": "exact",
        "mandatory": True,
        "type": "string",
    },
    {
        "id": "contracting_authority_address",
        "group": "A",
        "path": ["Contracting Authority", "address"],
        "queries": [
            "Auftraggeber Adresse Straße PLZ Ort",
            "Vergabestelle Anschrift",
            "Postanschrift Auftraggeber",
            "Kontaktadresse Beschaffungsstelle",
        ],
        "instruction": "Full postal address of the contracting authority.",
        "extraction_mode": "exact",
        "mandatory": False,
        "type": "string",
    },

    # ── Group C: Timeline / Deadlines ────────────────────────────────────────
    {
        "id": "submission_deadline",
        "group": "C",
        "path": ["Submission Deadline"],
        "queries": [
            "Abgabefrist Angebot Datum Uhrzeit",
            "Einreichungsfrist Ende",
            "Angebotsfrist bis zum",
            "Fristablauf Angebotsabgabe",
        ],
        "metadata_filter": {"contains_deadline": True},
        "instruction": "Exact submission deadline date and time in German format (DD.MM.YYYY HH:MM).",
        "extraction_mode": "exact",
        "mandatory": True,
        "validation": "datetime_de",
        "type": "string",
    },
    {
        "id": "question_deadline",
        "group": "C",
        "path": ["Important Dates", "question_deadline"],
        "queries": [
            "Bieterfragen Frist Datum",
            "Rückfragen Termin",
            "Fragen Einreichung bis",
            "Biegerfragen letzte Möglichkeit",
        ],
        "metadata_filter": {"contains_deadline": True},
        "instruction": "Deadline for submitting questions/clarifications.",
        "extraction_mode": "exact",
        "mandatory": False,
        "type": "string",
    },

    # ── Group D: Contract Value ──────────────────────────────────────────────
    {
        "id": "estimated_value",
        "group": "D",
        "path": ["Contract Value", "estimated_value"],
        "queries": [
            "geschätzter Auftragswert EUR Netto",
            "Gesamtvolumen Rahmenvertrag Betrag",
            "Auftragswert Schätzung Vergabe",
            "Gesamtstunden Stundenvolumen Kalkulation Wert",
        ],
        "metadata_filter": {"contains_value": True},
        "instruction": (
            "Estimated total contract value in EUR. "
            "If expressed as calculation (e.g. hours × rate), compute it and return numeric result."
        ),
        "extraction_mode": "numeric",
        "mandatory": False,
        "type": "number",
    },

    # ── Group F: Eligibility ─────────────────────────────────────────────────
    {
        "id": "required_certifications",
        "group": "F",
        "path": ["Scope & Requirements", "Standards & Certifications"],
        "queries": [
            "Zertifizierungen Qualifikationen gefordert",
            "Eignungskriterien Nachweise Zertifikat",
            "IPMA Prince2 ITIL Zertifizierung",
            "fachliche Qualifikation Anforderung",
        ],
        "instruction": "List all required or preferred certifications and qualifications.",
        "extraction_mode": "list",
        "mandatory": False,
        "type": "array",
    },

    # ── Group G: Document Checklist ──────────────────────────────────────────
    # NOTE: This group is handled by document_extractor.py separately
    # Include here only for audit trail / schema completeness
    {
        "id": "required_documents",
        "group": "G",
        "path": ["Supplier Eligibility", "Offer Submission Documents"],
        "queries": DOCUMENT_QUERIES if False else [  # populated by document_extractor
            "einzureichende Unterlagen Formblätter",
            "Angebot beizufügende Nachweise",
        ],
        "instruction": (
            "Complete list of ALL documents to be submitted with the bid. "
            "Include every form, declaration, certificate, and annex. "
            "COMPLETENESS IS CRITICAL."
        ),
        "extraction_mode": "list",
        "mandatory": True,
        "metadata_filter": {"contains_document_list": True},
        "type": "array",
        "use_dedicated_extractor": True,  # flag to route to document_extractor.py
    },

    # ── Group H: Personnel (per lot) ─────────────────────────────────────────
    {
        "id": "personnel_profiles_lot1",
        "group": "H",
        "path": ["Technical & Professional Ability", "Los 1"],
        "queries": [
            "Los 1 Projektmanager Anforderungen Profil",
            "Los 1 Qualifikation Erfahrung gefordert",
            "Projektmanager Junior Anforderungen",
            "Los 1 Leistungsbeschreibung Mitarbeiter",
        ],
        "metadata_filter": {"lot_scope": {"$in": ["[1]", "[1,2,3]", "all"]}},
        "instruction": "Required qualifications, experience, and certifications for Los 1 (Projektmanager).",
        "extraction_mode": "narrative",
        "mandatory": False,
        "type": "string",
    },
    {
        "id": "personnel_profiles_lot2",
        "group": "H",
        "path": ["Technical & Professional Ability", "Los 2"],
        "queries": [
            "Los 2 Projektmanager Senior Anforderungen",
            "Los 2 Senior Profil Qualifikation",
            "Projektmanager Senior Erfahrung gefordert",
            "Los 2 Programm-Management Anforderung",
        ],
        "metadata_filter": {"lot_scope": {"$in": ["[2]", "[1,2,3]", "all"]}},
        "instruction": "Required qualifications, experience, and certifications for Los 2 (Projektmanager Senior).",
        "extraction_mode": "narrative",
        "mandatory": False,
        "type": "string",
    },
    {
        "id": "personnel_profiles_lot3",
        "group": "H",
        "path": ["Technical & Professional Ability", "Los 3"],
        "queries": [
            "Los 3 Projektassistenz Anforderungen",
            "Los 3 Assistenz Profil Qualifikation",
            "Projektassistenz Erfahrung Anforderung",
            "Los 3 Unterstützung Projektleitung",
        ],
        "metadata_filter": {"lot_scope": {"$in": ["[3]", "[1,2,3]", "all"]}},
        "instruction": "Required qualifications, experience, and certifications for Los 3 (Projektassistenz).",
        "extraction_mode": "narrative",
        "mandatory": False,
        "type": "string",
    },

    # ── Group I: Award Criteria ───────────────────────────────────────────────
    {
        "id": "price_weight_percent",
        "group": "I",
        "path": ["Award Criteria", "price_weight_percent"],
        "queries": [
            "Preispunkte Gewichtung Prozent",
            "Preis Anteil Wertung Prozent",
            "Preisbewertung Gewicht",
            "40 Prozent Preis Kennzahl",
        ],
        "metadata_filter": {"$or": [{"content_type": "table"}, {"content_type": "list"}]},
        "instruction": "Percentage weight of price in the award criteria (e.g. 40).",
        "extraction_mode": "numeric",
        "mandatory": False,
        "validation": "percentage",
        "type": "number",
    },
    {
        "id": "quality_weight_percent",
        "group": "I",
        "path": ["Award Criteria", "quality_weight_percent"],
        "queries": [
            "Leistungspunkte Gewichtung Prozent",
            "Qualität Anteil Wertung",
            "Leistungsbewertung Gewicht Prozent",
            "60 Prozent Leistung Kennzahl",
        ],
        "metadata_filter": {"$or": [{"content_type": "table"}, {"content_type": "list"}]},
        "instruction": "Percentage weight of quality/performance in the award criteria (e.g. 60).",
        "extraction_mode": "numeric",
        "mandatory": False,
        "validation": "percentage",
        "type": "number",
    },
]

# Pre-defined queries for document checklist (used by document_extractor.py)
DOCUMENT_QUERIES = [
    "einzureichende Unterlagen Angebot Nachweise",
    "Formblätter Anlagen beizufügen",
    "Eignungsnachweis vorzulegen einreichen",
    "Teilnahmebedingungen Unterlagen Pflicht",
    "Angebotsbestandteile Checkliste vollständig",
    "Formblatt E1 E2 E3 E4 E5 E6 Mantelbogen",
    "Pflichtangaben Bieter Eigenerklärung",
    "Handelsregister Betriebshaftpflicht Versicherung Nachweis",
]

# Fields where null is unacceptable — trigger recovery pass
MANDATORY_FIELDS = [
    "contracting_authority_name",
    "contract_title",
    "submission_deadline",
    "procedure_type",
    "required_documents",
]
```

---

## 10. Testing & Evaluation Framework

### 10.1 Metrics

| Metric | Target | Notes |
|---|---|---|
| **Field Fill Rate** | >95% | Track per-field and per-group |
| **Critical Field Rate** | 100% | `submission_deadline`, `contracting_authority`, `procedure_type` |
| **Document Recall** | 100% | `# docs found / # docs in gold standard` |
| **Hallucination Rate** | <2% | Values not present in source text |
| **Null Classification Accuracy** | >90% | `confirmed_absent` vs `retrieval_miss` correct |

### 10.2 Ground Truth Annotation Format

```json
{
  "tender_id": "AS240020",
  "source_file": "ausschreibung_as240020.txt",
  "fields": {
    "submission_deadline": {
      "value": "16.08.2024 17:00",
      "chunk_indices": [167],
      "section": "Fristen"
    },
    "required_documents": {
      "value": ["EEE", "Mantelbogen", "Formblatt E1", "..."],
      "chunk_indices": [98, 109, 151, 152, 247, 299],
      "section": "Eignungsnachweise"
    }
  }
}
```

### 10.3 Evaluator

```python
# evaluation/evaluator.py

def evaluate(extracted: dict, gold: dict) -> dict:
    results = {"fill_rate": 0, "field_scores": {}, "doc_recall": 0}
    total, filled, correct = 0, 0, 0

    for field, gold_data in gold["fields"].items():
        total += 1
        extracted_val = extracted.get(field)
        gold_val = gold_data["value"]

        if extracted_val is not None:
            filled += 1
            if _values_match(extracted_val, gold_val):
                correct += 1
                results["field_scores"][field] = "correct"
            else:
                results["field_scores"][field] = f"wrong: got={extracted_val!r} expected={gold_val!r}"
        else:
            null_type = extracted.get(f"{field}__null_type", "unknown")
            results["field_scores"][field] = f"null ({null_type})"

    results["fill_rate"]    = filled / total if total else 0
    results["accuracy"]     = correct / total if total else 0

    # Document recall
    gold_docs  = set(d.lower() for d in (gold["fields"].get("required_documents", {}).get("value", [])))
    found_docs = set(d.get("name", "").lower() for d in (extracted.get("required_documents") or []))
    results["doc_recall"] = len(gold_docs & found_docs) / len(gold_docs) if gold_docs else 1.0

    return results


def _values_match(extracted, gold) -> bool:
    if isinstance(gold, list):
        gold_set = {str(v).lower().strip() for v in gold}
        if isinstance(extracted, list):
            ext_set = {str(v).lower().strip() for v in extracted}
        else:
            ext_set = {str(extracted).lower().strip()}
        # Partial match: at least 80% of gold items found
        return len(gold_set & ext_set) / len(gold_set) >= 0.8 if gold_set else True
    return str(extracted).strip().lower() == str(gold).strip().lower()
```

### 10.4 Ablation Test Order

Run these in sequence to measure marginal impact of each improvement:

1. **Baseline** — current system, measure fill rate on 5 tenders
2. **+BM25 hybrid retrieval** — expect +5–10% on exact-match fields
3. **+Query expansion** (4 queries per field) — expect +10–15% overall
4. **+LLM metadata enrichment** + filtered retrieval — expect +5–8%
5. **+Dedicated document extractor** — expect +15–20% on document fields
6. **+Agentic null recovery** — expect +5–10% on residual nulls
7. **+Cross-encoder reranking** — expect +3–5% on multi-section fields
8. **+Consistency checks** — quality improvement, not fill rate

---

## 11. Implementation Roadmap

### Phase 1 — Highest Impact (implement first)

| Component | Key Changes | Expected Gain |
|---|---|---|
| **BM25 Hybrid Search** | Add `rank-bm25`, implement `rrf_merge()`, replace `similarity_search` | +10–15% |
| **Query Expansion** | 4 German queries per field via LLM, deduplicated RRF fusion | +10–15% |

### Phase 2 — High Impact

| Component | Key Changes | Expected Gain |
|---|---|---|
| **LLM Metadata Enrichment** | Index-time: `content_type`, `lot_scope`, `contains_deadline` per chunk | +5–8% |
| **Dedicated Document Extractor** | 8 queries + 3 passes + list mode + standard forms cross-check | +15–20% on docs |

### Phase 3 — Medium Impact

| Component | Key Changes | Expected Gain |
|---|---|---|
| **Revised Field Grouping** | 10 semantic groups, per-lot extraction for personnel (3 calls) | +5–8% |
| **Null Classifier + Recovery Agent** | 3-type null detection, targeted re-retrieval for type 2/3 | +5–10% |

### Phase 4 — Quality & Robustness

| Component | Key Changes | Expected Gain |
|---|---|---|
| **Cross-Encoder Reranking** | `ms-marco-MiniLM-L-6-v2`, reduce context to top-25 chunks | +3–5% |
| **Consistency Checks** | Deadline ordering, weight sums, CPV format, lot calculations | Quality only |

---

## 12. New File Structure

```
tender_rag/
├── main.py
├── config.py                         # REVISED — extended FIELDS schema
│
├── ingestion/
│   ├── chunker.py                    # REVISED — structural anchoring
│   ├── table_reconstructor.py        # NEW — table detection & reconstruction
│   └── metadata_enricher.py          # NEW — LLM-based chunk metadata
│
├── vectorstore/
│   ├── store.py                      # REVISED — hybrid_retrieve()
│   ├── bm25_index.py                 # NEW — BM25 build, search, persist
│   └── fusion.py                     # NEW — RRF merge function
│
├── extraction/
│   ├── prompts.py                    # REVISED — extraction mode hints
│   ├── extractor.py                  # REVISED — structured output, grouped
│   ├── query_expander.py             # NEW — multi-query generation
│   ├── multi_query_retriever.py      # NEW — multi-query + dedup
│   ├── document_extractor.py         # NEW — dedicated doc checklist pipeline
│   ├── reranker.py                   # NEW — cross-encoder reranking
│   └── validator.py                  # REVISED — extended Pydantic schema
│
├── audit/
│   ├── null_classifier.py            # NEW — 3-type null detection
│   ├── recovery_agent.py             # NEW — targeted re-retrieval
│   ├── consistency_checks.py         # NEW — post-extraction logic validation
│   └── forms_validator.py            # NEW — standard forms cross-check
│
├── output/
│   └── writer.py                     # REVISED — null_type annotations
│
├── evaluation/
│   ├── evaluator.py                  # NEW — fill rate + recall metrics
│   └── gold_standard/               # NEW — annotated ground-truth JSONs
│       └── AS240020_gold.json
│
├── data/tenders/
├── schemas/tender_schema.json
└── requirements.txt                  # REVISED (see below)
```

### Updated requirements.txt

```txt
langchain>=0.2.0
langchain-google-genai>=1.0.0
langchain-chroma>=0.1.0
chromadb>=0.5.0
pydantic>=2.0.0
python-dotenv>=1.0.0
google-generativeai>=0.7.0
rank-bm25>=0.2.2
sentence-transformers>=2.7.0
```

---

## Critical Questions to Resolve Before Implementation

> Answer these before writing code — they affect architectural decisions.

**Q1: Are your `.txt` files generated from PDF or native text?**
→ If PDF-origin: table reconstruction is your #1 priority. Most null values in `Contract Volume`, `Award Criteria`, and `Document Lists` are caused by flattened tables.

**Q2: Do your tenders follow EU Official Journal / TED format?**
→ If yes: TED section codes (`II.1.4`, `III.1.1`, `IV.2.2`) can be used as hard metadata anchors instead of regex-detected headings — far more reliable.

**Q3: What is your acceptable latency per tender?**
→ LLM metadata enrichment adds ~30–60s at index time but pays off on every query. If re-indexing on each run is unacceptable, persist the enriched ChromaDB collection.

**Q4: Are `Personnel Profiles` in the tender document or submitted by bidders?**
→ If profiles are submitted in bids, they will **always** be null in the tender extraction — by design. Confirm this before treating it as a bug.

**Q5: Do you process multiple tenders in parallel?**
→ BM25 index is per-tender. ChromaDB collection should be namespaced per tender (`collection_name=tender_filename`). Do not share indices across tenders.