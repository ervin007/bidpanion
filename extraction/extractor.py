import os
from pydantic import BaseModel, Field
from typing import Optional, Literal
from langchain_google_vertexai import ChatVertexAI
from langchain_core.prompts import ChatPromptTemplate
from langfuse.langchain import CallbackHandler
from config import VERTEX_PROJECT_ID, VERTEX_LOCATION, LLM_MODEL, ROUTING_RULES

from typing import Optional, Literal, Any

class FieldExtraction(BaseModel):
    value: Any = Field(default=None, description="The extracted value, string or list, or null if not found")
    chunk_index: Optional[list[int]] = Field(default=None, description="List of chunk indices where the information was found")
    status: Literal["found", "confirmed_absent", "retrieval_gap"] = Field(description="Status of extraction")

def get_llm():
    return ChatVertexAI(
        model_name=LLM_MODEL,
        project=VERTEX_PROJECT_ID,
        location=VERTEX_LOCATION,
        temperature=0.0,
        max_retries=1,
        timeout=60
    )

def extract_field(field_cfg: dict, context_chunks: list, callbacks: list = None, tags: list = None) -> dict:
    llm = get_llm()
    structured_llm = llm.with_structured_output(FieldExtraction)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Du bist ein hochpräziser Analyst für deutsche Ausschreibungsunterlagen. 
Deine Aufgabe ist es, Informationen VOLLSTÄNDIG und im ORIGINALWORTLAUT zu extrahieren. 
Fasse niemals zusammen. Wenn mehrere Informationen (z.B. mehrere Termine oder Anforderungen) im Kontext stehen, extrahiere sie ALLE. 
Wenn der Kontext Tabellen oder Listen enthält, gib diese strukturiert wieder.
Antworte in der Sprache des Kontextes (Deutsch).

{routing_rules}"""),
        ("human", """Hier ist der Kontext aus den Vergabeunterlagen:
----------
{context}
----------

Extrahiere die Informationen für das folgende Feld: {query}
Spezifische Anweisung: {instruction}

WICHTIG: Sei so detailliert wie möglich. Wenn du die Information nicht findest, gib den Status 'confirmed_absent' oder 'retrieval_gap' an.""")
    ])
    
    import time
    max_retries = 5
    current_chunks = context_chunks[:]
    
    for attempt in range(max_retries):
        context_str = "\n\n".join([f"[Chunk {d.metadata.get('chunk_index', '?')}]\n{d.page_content}" for d in current_chunks])
        chain = prompt | structured_llm
        
        try:
            result = chain.invoke({
                "routing_rules": ROUTING_RULES,
                "context": context_str,
                "query": field_cfg["queries"][0], # Use primary query for extraction instructions
                "instruction": field_cfg["instruction"]
            }, config={"callbacks": callbacks or [], "tags": tags or []})
            res_dict = result.dict()
            if res_dict.get("value") is None:
                if len(current_chunks) > 5:
                    new_len = int(len(current_chunks) * 0.7)
                    print(f"\n[Extraction Null] Value was null for {field_cfg['id']}. Reducing context from {len(current_chunks)} to {new_len} chunks and retrying... (Attempt {attempt + 1}/{max_retries})")
                    current_chunks = current_chunks[:new_len]
                    continue
            return res_dict
        except Exception as e:
            err_msg = str(e).lower()
            if any(x in err_msg for x in ["429", "resource exhausted", "quota"]):
                import random
                wait_time = 5 + random.uniform(1, 5) # Add jitter
                print(f"\n[Rate Limit] Hit API limits for {field_cfg['id']}. Retrying in {wait_time:.1f}s... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
            else:
                if len(current_chunks) > 5:
                    new_len = int(len(current_chunks) * 0.7)
                    print(f"\n[Extraction Error] {e} for {field_cfg['id']}. Reducing context from {len(current_chunks)} to {new_len} chunks and retrying... (Attempt {attempt + 1}/{max_retries})")
                    current_chunks = current_chunks[:new_len]
                else:
                    print(f"\nExtraction API permanently failed for field {field_cfg['id']}: {e}")
                    return {"value": None, "chunk_index": None, "status": "retrieval_gap"}
                
    print(f"\nExtraction permanently failed for field {field_cfg['id']} after {max_retries} retries.")
    return {"value": None, "chunk_index": None, "status": "retrieval_gap"}
