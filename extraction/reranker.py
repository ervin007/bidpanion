import os
import json
from langchain_core.documents import Document
from langchain_google_vertexai import ChatVertexAI
from langchain_core.prompts import ChatPromptTemplate
from langfuse.langchain import CallbackHandler
from config import VERTEX_PROJECT_ID, VERTEX_LOCATION, LLM_MODEL

def rerank(query: str, docs: list[Document], top_k: int = 25, callbacks: list = None, tags: list = None) -> list[Document]:
    """
    Rerank retrieved chunks using Gemini 2.5 via Vertex AI.
    This replaces the local cross-encoder with a smarter LLM-based evaluator.
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
    
    # Format the documents into a numbered list
    docs_text = "\n\n".join([f"[{i}] {d.page_content}" for i, d in enumerate(docs)])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Du bist ein Experte für Informationsabruf. Deine Aufgabe ist es, Textabschnitte nach ihrer Relevanz für eine gegebene Suchanfrage zu bewerten."),
        ("human", """Bewerte die folgenden Textabschnitte basierend darauf, wie gut sie die Suchanfrage beantworten.
        
Suchanfrage: {query}

Textabschnitte:
{docs_text}

Gib AUSSCHLIESSLICH ein JSON-Array mit den Indizes der {top_k} relevantesten Abschnitte zurück, sortiert nach absteigender Relevanz.
Beispiel: [12, 4, 0, 45, 8]""")
    ])
    
    chain = prompt | llm
    
    import time
    
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = chain.invoke({
                "query": query,
                "docs_text": docs_text,
                "top_k": min(top_k, len(docs))
            }, config={"callbacks": callbacks or [], "tags": tags or []})
            
            import re
            
            # Safely extract the JSON array using regex in case the LLM adds conversational text
            content = response.content.strip()
            match = re.search(r'\[\s*\d+(?:\s*,\s*\d+)*\s*\]', content)
            if match:
                ranked_indices = json.loads(match.group(0))
            else:
                # Fallback if regex fails, maybe it's just pure JSON
                cleaned = content.replace("```json", "").replace("```", "").strip()
                ranked_indices = json.loads(cleaned)
            
            # Return the documents in the order specified by Gemini
            ranked_docs = []
            for idx in ranked_indices:
                if isinstance(idx, int) and 0 <= idx < len(docs):
                    ranked_docs.append(docs[idx])
                    
            # If Gemini didn't return enough, pad with the original RRF order
            if len(ranked_docs) < min(top_k, len(docs)):
                seen = set(ranked_indices)
                for i, doc in enumerate(docs):
                    if i not in seen:
                        ranked_docs.append(doc)
                        if len(ranked_docs) >= min(top_k, len(docs)):
                            break
                            
            return ranked_docs
        except Exception as e:
            if "429" in str(e) or "Resource exhausted" in str(e) or "quota" in str(e).lower():
                wait_time = 60 * (attempt + 1)
                print(f"\n[Rate Limit] Reranker hit API limits. Retrying in {wait_time}s... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
            else:
                print(f"LLM Reranking failed: {e}. Falling back to original RRF order.")
                return docs[:top_k]
                
    print(f"\nReranking permanently failed after {max_retries} retries. Falling back to original RRF order.")
    return docs[:top_k]
