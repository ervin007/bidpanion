import os
import json
import logging
import asyncio
import httpx
from temporalio import activity

from ingestion.chunker import load_and_chunk
from vectorstore.bm25_index import build_bm25
from vectorstore.store import build_vectorstore, hybrid_retrieve
from extraction.reranker import rerank
from extraction.extractor import extract_field
from config import FIELDS, TOP_K_DENSE, TOP_K_SPARSE, TOP_K_FINAL, TOP_K_RERANK
import datetime
from langfuse.langchain import CallbackHandler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExtractionActivities:
    def __init__(self):
        self.langfuse_handler = None
        if os.environ.get("LANGFUSE_PUBLIC_KEY"):
            self.langfuse_handler = CallbackHandler()
        
        # In-memory cache for indices
        self._bm25_index = None
        self._bm25_docs = None
        self._vectorstore = None
        self._current_file = None
        self._index_lock = asyncio.Lock()

    @activity.defn
    async def ingest_document(self, input_file: str) -> dict:
        logger.info(f"Ingesting document: {input_file}")
        documents = await asyncio.to_thread(load_and_chunk, input_file)
        return {"count": len(documents), "file": input_file}

    @activity.defn
    async def prepare_indices(self, input_file: str) -> bool:
        async with self._index_lock:
            if self._current_file == input_file and self._vectorstore is not None:
                logger.info(f"Indices already prepared for {input_file}, skipping rebuild.")
                return True
                
            logger.info(f"Preparing indices for: {input_file}")
            documents = await asyncio.to_thread(load_and_chunk, input_file)
            # Build and cache
            self._bm25_index, self._bm25_docs = await asyncio.to_thread(build_bm25, documents)
            self._vectorstore = await asyncio.to_thread(build_vectorstore, documents)
            self._current_file = input_file
            return True

    @activity.defn
    async def extract_field_activity(self, field_id: str, input_file: str) -> dict:
        logger.info(f"Extracting field: {field_id}")
        
        # Ensure indices are loaded
        if self._current_file != input_file or self._vectorstore is None:
            logger.info(f"Indices not cached for {input_file}, building now...")
            await self.prepare_indices(input_file)

        # Find field definition
        field = next((f for f in FIELDS if f["id"] == field_id), None)
        if not field:
            raise ValueError(f"Field {field_id} not found in config")

        run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        field_tags = [f"run_{run_timestamp}", f"Field: {field['id']}"]
        
        all_retrieved = []
        for query in field["queries"]:
            docs = await asyncio.to_thread(
                hybrid_retrieve,
                self._vectorstore, self._bm25_index, self._bm25_docs, query,
                dense_k=TOP_K_DENSE, sparse_k=TOP_K_SPARSE, final_k=TOP_K_FINAL,
                metadata_filter={"source": input_file}
            )
            all_retrieved.extend(docs)

            
        unique_docs = {d.metadata["chunk_index"]: d for d in all_retrieved}
        top_fusion = list(unique_docs.values())
        
        callbacks = [self.langfuse_handler] if self.langfuse_handler else None
        
        reranked_docs = await asyncio.to_thread(rerank, field["queries"][0], top_fusion, top_k=TOP_K_RERANK, callbacks=callbacks, tags=field_tags)
        result = await asyncio.to_thread(extract_field, field, reranked_docs, callbacks=callbacks, tags=field_tags)
        
        return {
            "field_id": field_id,
            "result": result,
            "path": field["path"],
            "type": field.get("type")
        }

    @activity.defn
    async def save_final_results(self, results: list, output_file: str) -> str:
        logger.info(f"Saving final results to {output_file}")
        
        def set_nested_value(d, path, value):
            for key in path[:-1]:
                d = d.setdefault(key, {})
            d[path[-1]] = value

        final_output = {
            "Contracting Authority": None,
            "Project Description": None,
            "Submission Deadline": None,
            "Important Dates": None,
            "Scope & Requirements": {
                "Scope & Requirements": None,
                "Contract Volume": None,
                "Place of Performance": None,
                "Standards & Certifications": [],
                "Subcontracting & Consortia": None,
                "Forms & e-Submission": None
            },
            "Supplier Eligibility": {
                "Offer Submission Documents": [],
                "List of Documents": None,
                "Economic & Financial Standing": {
                    "Minimum Turnover": None,
                    "Turnover in Comparable Services": None
                },
                "Legal & Registration": {
                    "Trade/Professional Register Entry": None,
                    "Self-declarations (GWB §§123/124 or equivalent)": [],
                    "Subcontractor Identification & Reliance": None
                }
            },
            "Technical & Professional Ability": {
                "Personnel Profiles": None,
                "Headcount / Staffing": None,
                "Reference Projects": None
            },
            "Company Referrals": None,
            "Award Criteria": None
        }

        citations = {"sources": {}}
        warnings = []

        for item in results:
            field_id = item["field_id"]
            result = item["result"]
            path = item["path"]
            field_type = item["type"]

            extracted_val = result.get("value")
            if extracted_val is not None:
                if field_type == "array" and not isinstance(extracted_val, list):
                    extracted_val = [x.strip() for x in str(extracted_val).split(",") if x.strip()]
                set_nested_value(final_output, path, extracted_val)

            # Citations
            source_key = f"{path[-1]}__quelle"
            chunk_idx = result.get("chunk_index")
            if isinstance(chunk_idx, list):
                citations["sources"][source_key] = ", ".join(map(str, chunk_idx)) if chunk_idx else None
            else:
                citations["sources"][source_key] = str(chunk_idx) if chunk_idx is not None else None

            # Warnings
            from config import MANDATORY_FIELDS
            if field_id in MANDATORY_FIELDS and result.get("status") != "found":
                warnings.append(f"Mandatory field missing: {field_id} (Reason: {result.get('status')})")

        final_output["citations"] = citations
        if warnings:
            final_output["citations"]["warnings"] = warnings

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(final_output, f, ensure_ascii=False, indent=2)
            
        return output_file

    @activity.defn
    async def send_completion_webhook_activity(self, callback_url: str, workflow_id: str, filename: str, status: str, output_file: str = None) -> bool:
        if not callback_url:
            return True
            
        payload = {
            "status": status,
            "workflow_id": workflow_id,
            "filename": filename
        }
        
        if output_file and os.path.exists(output_file):
            with open(output_file, 'r', encoding='utf-8') as f:
                payload["result"] = json.load(f)
                
        logger.info(f"Sending completion webhook to {callback_url}")
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(callback_url, json=payload)
            response.raise_for_status()
        return True

