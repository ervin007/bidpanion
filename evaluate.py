import os
import json
import logging
from datetime import datetime
from langchain_google_vertexai import ChatVertexAI
from langchain_core.prompts import ChatPromptTemplate
from langfuse.langchain import CallbackHandler
from config import VERTEX_PROJECT_ID, VERTEX_LOCATION

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%H:%M:%S')

def evaluate_extraction():
    # 1. Load the manual validation data (The Reference)
    validation_path = "validation.txt"
    if not os.path.exists(validation_path):
        logging.error(f"Validation file {validation_path} not found.")
        return

    with open(validation_path, "r", encoding="utf-8") as f:
        validation_data = f.read()

    # 2. Load the AI extraction result (The Candidate)
    # Using the path specified by the user
    result_path = "output/result_new_config_v2.json"
    if not os.path.exists(result_path):
        logging.error(f"Result file {result_path} not found.")
        return

    with open(result_path, "r", encoding="utf-8") as f:
        ai_result = json.load(f)

    logging.info(f"Starting Forensic Audit: {result_path} vs {validation_path}")

    # 3. Setup the Judge (Gemini 1.5 Pro)
    # We use a high-intelligence model for the evaluation step
    llm = ChatVertexAI(
        model_name="gemini-2.5-pro",
        project=VERTEX_PROJECT_ID,
        location=VERTEX_LOCATION,
        temperature=0.0
    )

    # 4. Define the Audit Prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Du bist ein Senior-Prüfer für Vergabeunterlagen. 
Deine Aufgabe ist ein forensisches Audit zwischen einer manuellen Referenz (Ground Truth) und einer KI-Extraktion (JSON).

PRÜF-KRITERIEN:
1. KORREKTHEIT: Entsprechen die Fakten im JSON der Referenz?
2. VOLLSTÄNDIGKEIT: Wurden alle wichtigen Details aus der Referenz im JSON erfasst?
3. HALLUZINATIONEN: Enthält das JSON Informationen, die weder in der Referenz noch im Originaltext stehen?

DEIN OUTPUT:
Erstelle einen detaillierten Markdown-Bericht. Markiere Diskrepanzen klar mit:
- ❌ FEHLER: Falsche Information.
- ⚠️ FEHLEND: Information in der Referenz vorhanden, aber nicht im JSON.
- ✅ KORREKT: Erfolgreich abgeglichene Felder.

Sei streng und präzise."""),
        ("human", """
### MANUELLE REFERENZ (Bodenwahrheit):
{reference}

### KI-EXTRAKTION (JSON):
{candidate}

---
Führe das Audit durch und erstelle den Bericht:""")
    ])

    # 5. Langfuse Integration
    # This allows you to see the "Judge's" reasoning in your dashboard
    langfuse_handler = CallbackHandler()

    # 6. Run the Audit
    chain = prompt | llm
    
    logging.info("Running comparison through Gemini 2.5 Pro...")
    
    session_id = f"audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    response = chain.invoke(
        {"reference": validation_data, "candidate": json.dumps(ai_result, indent=2, ensure_ascii=False)},
        config={
            "callbacks": [langfuse_handler],
            "tags": ["Evaluation", "Forensic Audit"],
            "metadata": {"session_id": session_id}
        }
    )

    # 7. Save and Output the Report
    report_path = "output/audit_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(response.content)

    logging.info(f"Audit complete! Report saved to: {report_path}")
    print("\n" + "="*50)
    print("AUDIT SUMMARY PREVIEW:")
    print("="*50)
    # Print a small snippet of the report to the terminal
    print(response.content[:1000] + "...")
    print("="*50)

if __name__ == "__main__":
    evaluate_extraction()
