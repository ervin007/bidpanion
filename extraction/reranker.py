import os
import json
from langchain_core.documents import Document
from langchain_google_vertexai import ChatVertexAI
from langchain_core.prompts import ChatPromptTemplate
from langfuse.langchain import CallbackHandler
from config import VERTEX_PROJECT_ID, VERTEX_LOCATION, LLM_MODEL, TOP_K_FINAL

def rerank(query: str, docs: list[Document], top_k: int = 25, callbacks: list = None, tags: list = None) -> list[Document]:
    """
    Rerank retrieved chunks using Gemini 2.5 via Vertex AI.
    Optimized to handle large contexts by capping input and using previews.
    """
    if not docs:
        return docs
        
    llm = ChatVertexAI(
        model_name=LLM_MODEL,
        project=VERTEX_PROJECT_ID,
        location=VERTEX_LOCATION,
        temperature=0.0,
        max_retries=1,
        timeout=60
    )
    
    import time
    max_retries = 5
    current_max_docs = TOP_K_FINAL # Use all retrieved docs initially
    preview_length = 1000
    
    # Define template once to avoid parsing errors in the loop
    human_template = """Bewerte die folgenden Textabschnitte basierend darauf, wie gut sie die Suchanfrage beantworten.
            
Suchanfrage: {query}

Textabschnitte:
{docs_text}

Gib AUSSCHLIESSLICH ein JSON-Array mit den Indizes der {top_k} relevantesten Abschnitte zurück, sortiert nach absteigender Relevanz.
Beispiel: [12, 4, 0, 45, 8]"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Du bist ein Experte für Informationsabruf. Deine Aufgabe ist es, Textabschnitte nach ihrer Relevanz für eine gegebene Suchanfrage zu bewerten."),
        ("human", human_template)
    ])
    
    chain = prompt | llm

    for attempt in range(max_retries):
        input_docs = docs[:current_max_docs]
        
        # Format the documents into a numbered list
        docs_text = ""
        for i, d in enumerate(input_docs):
            preview = d.page_content[:preview_length].replace("\n", " ").replace("{", "(").replace("}", ")")
            docs_text += f"[{i}] {preview}...\n\n"
        
        try:
            response = chain.invoke({
                "query": query,
                "docs_text": docs_text,
                "top_k": min(top_k, len(input_docs))
            }, config={"callbacks": callbacks or [], "tags": tags or []})
            
            import re
            content = response.content.strip()
            # Use regex to find the JSON array in case of extra text
            match = re.search(r'\[\s*\d+(?:\s*,\s*\d+)*\s*\]', content)
            if match:
                ranked_indices = json.loads(match.group(0))
            else:
                # Fallback to direct load
                cleaned = content.replace("```json", "").replace("```", "").strip()
                ranked_indices = json.loads(cleaned)
            
            ranked_docs = []
            for idx in ranked_indices:
                if isinstance(idx, int) and 0 <= idx < len(input_docs):
                    ranked_docs.append(input_docs[idx])
                    
            if len(ranked_docs) < min(top_k, len(docs)):
                seen_ids = {id(d) for d in ranked_docs}
                for doc in docs:
                    if id(doc) not in seen_ids:
                        ranked_docs.append(doc)
                        if len(ranked_docs) >= min(top_k, len(docs)):
                            break
                            
            return ranked_docs
        except Exception as e:
            err_msg = str(e).lower()
            if any(x in err_msg for x in ["429", "resource exhausted", "quota"]):
                import random
                wait_time = 5 + random.uniform(1, 5) # Add jitter
                # Reduce tokens on rate limit too (TPM management)
                current_max_docs = max(5, int(current_max_docs * 0.6))
                preview_length = max(200, int(preview_length * 0.7))
                print(f"\n[Rate Limit] Reranker hit API limits. Waiting {wait_time:.1f}s and reducing load (docs: {current_max_docs}, preview: {preview_length})...")
                time.sleep(wait_time)
            else:
                current_max_docs = max(5, int(current_max_docs * 0.5))
                preview_length = max(200, int(preview_length * 0.5))
                print(f"\n[Reranker Error] {e}. Reducing input to {current_max_docs} docs and {preview_length} chars, retrying (Attempt {attempt + 1}/{max_retries})...")
                
    print(f"\nReranking permanently failed after {max_retries} retries. Falling back to original RRF order.")
    return docs[:top_k]
