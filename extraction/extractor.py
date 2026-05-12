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
        ("system", f"""Du bist ein hochpräziser Analyst für deutsche Ausschreibungsunterlagen. 
Deine Aufgabe ist es, Informationen VOLLSTÄNDIG und im ORIGINALWORTLAUT zu extrahieren. 
Fasse niemals zusammen. Wenn mehrere Informationen (z.B. mehrere Termine oder Anforderungen) im Kontext stehen, extrahiere sie ALLE. 
Wenn der Kontext Tabellen oder Listen enthält, gib diese strukturiert wieder.
Antworte in der Sprache des Kontextes (Deutsch).

{ROUTING_RULES}"""),
        ("human", """Hier ist der Kontext aus den Vergabeunterlagen:
----------
{context}
----------

Extrahiere die Informationen für das folgende Feld: {query}
Spezifische Anweisung: {instruction}

WICHTIG: Sei so detailliert wie möglich. Wenn du die Information nicht findest, gib den Status 'confirmed_absent' oder 'retrieval_gap' an.""")
    ])
    
    context_str = "\n\n".join([f"[Chunk {d.metadata.get('chunk_index', '?')}]\n{d.page_content}" for d in context_chunks])
    
    chain = prompt | structured_llm
    
    import time
    
    max_retries = 5
    for attempt in range(max_retries):
        try:
            result = chain.invoke({
                "context": context_str,
                "query": field_cfg["queries"][0], # Use primary query for extraction instructions
                "instruction": field_cfg["instruction"]
            }, config={"callbacks": callbacks or [], "tags": tags or []})
            return result.dict()
        except Exception as e:
            if "429" in str(e) or "Resource exhausted" in str(e) or "quota" in str(e).lower():
                wait_time = 60 * (attempt + 1)
                print(f"\n[Rate Limit] Hit API limits for {field_cfg['id']}. Retrying in {wait_time}s... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
            else:
                print(f"\nExtraction API failed for field {field_cfg['id']}: {e}")
                return {"value": None, "chunk_index": None, "status": "retrieval_gap"}
                
    print(f"\nExtraction permanently failed for field {field_cfg['id']} after {max_retries} retries.")
    return {"value": None, "chunk_index": None, "status": "retrieval_gap"}
