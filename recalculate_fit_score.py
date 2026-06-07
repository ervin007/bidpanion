import os
import json
import argparse
import asyncio
import httpx
from pydantic import BaseModel, Field
from typing import List, Literal
from extraction.extractor import get_llm
from langchain_core.prompts import ChatPromptTemplate

class FitCategoryEvaluation(BaseModel):
    slug: str = Field(description="Unique key of the category, matching company profile section slugs (e.g. services, industries, geography, certifications, capacity, etc.)")
    label: str = Field(description="Display label of the category")
    weight: int = Field(description="Percentage weight in overall score")
    score: int = Field(description="Evaluated score out of 100")
    status: Literal["MATCHED", "PARTIAL", "UNMATCHED", "NA"] = Field(description="Match status")
    details: str = Field(description="Explanation of evaluated score")
    matchedItems: List[str] = Field(description="List of matched features")
    unmatchedItems: List[str] = Field(description="List of missing requirements")

class FitScoreEvaluation(BaseModel):
    fitScore: int = Field(description="Overall Fit Score between 0 and 100")
    recommendation: Literal["BID", "REVIEW", "NO_BID"] = Field(description="Match quality recommendation")
    fitCategories: List[FitCategoryEvaluation] = Field(description="Evaluated category breakdown. Sum of category weights must equal 100.")

def extract_requirements_from_json(data: dict) -> str:
    """Recursively formats the output JSON dictionary into a flat string of requirements."""
    requirements = ""
    def recurse(d, prefix=""):
        if isinstance(d, dict):
            for k, v in d.items():
                if k == "citations":
                    continue
                recurse(v, f"{prefix} -> {k}" if prefix else k)
        elif isinstance(d, list):
            requirements += f"- {prefix}: {', '.join(map(str, d))}\n"
        elif d is not None:
            requirements += f"- {prefix}: {d}\n"
            
    recurse(data)
    return requirements

async def main():
    parser = argparse.ArgumentParser(description="Recalculate and backfill FitScore for a past tender.")
    parser.add_argument("--file", required=True, help="Path to the extracted output JSON file (e.g., output/tender.json)")
    parser.add_argument("--profile", required=True, help="Raw JSON string of the company profile")
    parser.add_argument("--callback", required=True, help="Frontend webhook callback URL (e.g., https://domain.com/api/ai/jobs/job_id/callback)")
    parser.add_argument("--workflow-id", default="recalculate", help="Workflow ID reference")
    parser.add_argument("--filename", default="recalculated.zip", help="Filename reference")
    args = parser.parse_args()

    # 1. Load the output JSON file
    if not os.path.exists(args.file):
        print(f"Error: File not found: {args.file}")
        return
        
    with open(args.file, "r", encoding="utf-8") as f:
        tender_data = json.load(f)

    requirements = extract_requirements_from_json(tender_data)
    print("Formatted Requirements:\n", requirements)

    # 2. Format the company profile
    profile_summary = ""
    try:
        profile_json = json.loads(args.profile)
        sections = profile_json.get("sections", [])
        for section in sections:
            label = section.get("label", "")
            data = section.get("data", {})
            profile_summary += f"### {label}\n"
            for k, v in data.items():
                profile_summary += f"- {k}: {v}\n"
    except Exception as pe:
        print(f"Failed to parse profile JSON: {pe}. Using raw profile string.")
        profile_summary = args.profile

    print("Formatted Company Profile:\n", profile_summary)

    # 3. Call Vertex AI
    print("Calling Vertex AI to calculate Fit Score...")
    llm = get_llm()
    structured_llm = llm.with_structured_output(FitScoreEvaluation)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """Du bist ein Experte für die Bewertung von Ausschreibungen (Go/No-Go-Entscheidungen).
Deine Aufgabe ist es, die Anforderungen einer Ausschreibung mit dem Firmenprofil (Company Profile) abzugleichen.
Berechne einen Gesamt-Fit-Score (0 bis 100) und gib eine Empfehlung ("BID", "REVIEW", oder "NO_BID") ab.
Teile die Bewertung in sinnvolle Kategorien auf, die auf den Abschnitten des Firmenprofils basieren (z. B. services, industries, geography, certifications, capacity, etc.).
Verteile die Gewichte (weights) so, dass sie in Summe genau 100 ergeben.
Antworte auf Deutsch für details, matchedItems und unmatchedItems."""),
        ("human", """Hier sind die Anforderungen aus der Ausschreibung:
----------
{requirements}
----------

Hier ist das Firmenprofil (Company Profile):
----------
{profile}
----------

Berechne den Fit Score.""")
    ])

    chain = prompt | structured_llm
    eval_result = chain.invoke({
        "requirements": requirements,
        "profile": profile_summary
    })

    fit_results = eval_result.dict()
    print("Fit Score Result:\n", json.dumps(fit_results, indent=2))

    # 4. Call Frontend callback
    payload = {
        "status": "completed",
        "workflow_id": args.workflow_id,
        "filename": args.filename,
        "result": tender_data,
        "fitScore": fit_results.get("fitScore"),
        "recommendation": fit_results.get("recommendation"),
        "fitCategories": fit_results.get("fitCategories")
    }

    headers = {}
    callback_token = os.environ.get("AI_PIPELINE_CALLBACK_TOKEN")
    if callback_token:
        headers["Authorization"] = f"Bearer {callback_token}"

    print(f"Sending callback to {args.callback}...")
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(args.callback, json=payload, headers=headers)
        print(f"Callback response: {response.status_code}")
        if response.status_code == 200:
            print("Successfully updated Fit Score in production database!")
        else:
            print(f"Failed callback response: {response.text}")

if __name__ == "__main__":
    asyncio.run(main())
