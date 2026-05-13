import os
import json
from ingestion.chunker import load_and_chunk
from vectorstore.bm25_index import build_bm25
from vectorstore.store import build_vectorstore, hybrid_retrieve
from extraction.reranker import rerank
from extraction.extractor import extract_field
from config import FIELDS, TOP_K_DENSE, TOP_K_SPARSE, TOP_K_FINAL, TOP_K_RERANK

def set_nested_value(d, path, value):
    for key in path[:-1]:
        d = d.setdefault(key, {})
    d[path[-1]] = value

def main():
    input_file = "input/2024_07_BG_Phoenics___RV_UL_im_Gescha_ftsbereich_PM__PS__Los_1_3.txt"
    output_file = "output/result.json"
    
    import logging
    logging.basicConfig(level=logging.INFO, format='\n[%(asctime)s] %(message)s', datefmt='%H:%M:%S')

    print(f"Loading and chunking {input_file}...")
    documents = load_and_chunk(input_file)
    print(f"Created {len(documents)} chunks.")
    
    print("Building BM25 index...")
    bm25_index, bm25_docs = build_bm25(documents)
    
    print("Building ChromaDB vectorstore...")
    vectorstore = build_vectorstore(documents)
    
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
    
    from tqdm import tqdm
    import datetime
    from langfuse.langchain import CallbackHandler
    
    # Initialize a single global Langfuse handler for the entire run
    langfuse_handler = None
    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if os.environ.get("LANGFUSE_PUBLIC_KEY"):
        os.environ["LANGFUSE_SESSION_ID"] = f"extraction_run_{run_timestamp}"
        langfuse_handler = CallbackHandler()
        logging.info("Langfuse tracing enabled with global session ID.")

    for field in tqdm(FIELDS, desc="Extracting tender data"):
        logging.info(f"=== STARTING FIELD: {field['id']} ===")
        
        # Create dynamic tags for this specific field + the unifying run tag
        field_tags = [f"run_{run_timestamp}", f"Field: {field['id']}"]
        
        all_retrieved = []
        logging.info("  [1/3] Running Hybrid Retrieval (Vector + BM25)...")
        for query in field["queries"]:
            docs = hybrid_retrieve(
                vectorstore, bm25_index, bm25_docs, query,
                dense_k=TOP_K_DENSE, sparse_k=TOP_K_SPARSE, final_k=TOP_K_FINAL
            )
            all_retrieved.extend(docs)
            
        unique_docs = {d.metadata["chunk_index"]: d for d in all_retrieved}
        top_fusion = list(unique_docs.values())
        
        logging.info(f"  [2/3] Reranking {len(top_fusion)} chunks with Gemini...")
        callbacks = [langfuse_handler] if langfuse_handler else None
        reranked_docs = rerank(field["queries"][0], top_fusion, top_k=TOP_K_RERANK, callbacks=callbacks, tags=field_tags)
        
        logging.info(f"  [3/3] Extracting final structure with Gemini...")
        result = extract_field(field, reranked_docs, callbacks=callbacks, tags=field_tags)
        
        # Build nested structure based on path
        extracted_val = result.get("value")
        if extracted_val is not None:
            # Type safety: if schema expects a list, ensure it's a list
            # Simple list casting if we know it's supposed to be an array
            if field.get("type") == "array" and not isinstance(extracted_val, list):
                extracted_val = [x.strip() for x in str(extracted_val).split(",") if x.strip()]
            
            set_nested_value(final_output, field["path"], extracted_val)
        
        # Add citations
        source_key = f"{field['path'][-1]}__quelle"
        chunk_idx = result.get("chunk_index")
        if isinstance(chunk_idx, list):
            citations["sources"][source_key] = ", ".join(map(str, chunk_idx)) if chunk_idx else None
        else:
            citations["sources"][source_key] = str(chunk_idx) if chunk_idx is not None else None
        
        # Track missing mandatory fields
        if field.get("mandatory") and result.get("status") != "found":
            warnings.append(f"Mandatory field missing: {field['id']} (Reason: {result.get('status')})")
            
        logging.info(f"=== FINISHED FIELD: {field['id']} ===")


    # Combine into final JSON format
    final_output["citations"] = citations
    if warnings:
        final_output["citations"]["warnings"] = warnings

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)
        
    print(f"Extraction complete. Results saved to {output_file}")

if __name__ == "__main__":
    main()
