import os
import json
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from temporalio.client import Client
from temporal.workflows import TenderExtractionWorkflow

app = FastAPI(title="Bidpanion Data API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

INPUT_DIR = "input"
OUTPUT_DIR = "output"
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

async def get_temporal_client():
    temporal_url = os.getenv("TEMPORAL_URL", "localhost:7233")
    return await Client.connect(temporal_url)

@app.post("/api/process")
async def process_tender(file: UploadFile = File(...), callback_url: str = Form(None)):
    """Uploads a tender zip file and starts the Temporal extraction workflow."""
    if not file.filename.endswith('.zip'):
        raise HTTPException(status_code=400, detail="Only .zip files are supported")
    
    input_path = os.path.join(INPUT_DIR, file.filename)
    content = await file.read()
    with open(input_path, "wb") as f:
        f.write(content)
        
    output_filename = os.path.splitext(file.filename)[0] + ".json"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    
    try:
        client = await get_temporal_client()
        workflow_id = f"extract-{os.path.splitext(file.filename)[0].replace(' ', '_')}"
        
        handle = await client.start_workflow(
            TenderExtractionWorkflow.run,
            args=[input_path, output_path, callback_url],
            id=workflow_id,
            task_queue="tender-extraction-queue",
        )
        return {
            "message": "Processing started",
            "workflow_id": handle.id,
            "filename": file.filename
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start workflow: {str(e)}")

@app.get("/api/status/{workflow_id}")
async def get_status(workflow_id: str):
    """Gets the current status of the extraction workflow."""
    try:
        client = await get_temporal_client()
        handle = client.get_workflow_handle(workflow_id)
        description = await handle.describe()
        status = description.status
        return {"workflow_id": workflow_id, "status": str(status.name)}
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Workflow not found or error: {str(e)}")

@app.get("/api/data")
async def list_documents():
    """Lists all available uploaded documents."""
    docs = [f for f in os.listdir(INPUT_DIR) if f.endswith('.txt')]
    return {"documents": docs}

@app.get("/api/file/{filename:path}", response_class=PlainTextResponse)
async def get_file(filename: str):
    """Retrieves the original txt file."""
    filepath = os.path.join(INPUT_DIR, filename)
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    raise HTTPException(status_code=404, detail="File not found")

@app.get("/api/results/{filename:path}")
async def get_results(filename: str):
    """Retrieves the extracted JSON results for a specific tender."""
    json_filename = os.path.splitext(filename)[0] + ".json"
    json_path = os.path.join(OUTPUT_DIR, json_filename)
    
    # Fallback if specific file isn't found (for backward compatibility)
    if not os.path.exists(json_path):
        json_path = os.path.join(OUTPUT_DIR, "result.json")

    if os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error reading results: {str(e)}")
            
    return {}

if __name__ == "__main__":
    print("Starting FastAPI Data Server at port 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
