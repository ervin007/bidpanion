import asyncio
import os
import sys
from temporalio.client import Client
from temporal.workflows import TenderExtractionWorkflow

async def main():
    # Connect to Temporal server
    # When running from host, use localhost:7233
    # When running inside docker, use temporal:7233
    temporal_url = os.environ.get("TEMPORAL_URL", "localhost:7233")
    client = await Client.connect(temporal_url)
    
    # Default paths or take from args
    input_file = sys.argv[1] if len(sys.argv) > 1 else "input/2024_07_BG_Phoenics___RV_UL_im_Gescha_ftsbereich_PM__PS__Los_1_3.txt"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "output/result.json"
    
    workflow_id = f"tender-extraction-{os.path.basename(input_file)}"
    
    print(f"Starting workflow {workflow_id}...")
    
    handle = await client.start_workflow(
        TenderExtractionWorkflow.run,
        args=[input_file, output_file],
        id=workflow_id,
        task_queue="bidpanion-task-queue",
    )
    
    print(f"Workflow started. ID: {handle.id}")
    print("Waiting for result (this may take several minutes)...")
    
    result = await handle.result()
    print(f"Workflow completed successfully!")
    print(f"Output saved to: {result['output']}")

if __name__ == "__main__":
    asyncio.run(main())
