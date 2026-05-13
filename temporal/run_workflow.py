import asyncio
import os
from temporalio.client import Client
from temporal.workflows import TenderExtractionWorkflow

async def main():
    # Connect to Temporal
    temporal_url = os.getenv("TEMPORAL_URL", "localhost:7233")
    client = await Client.connect(temporal_url)

    input_dir = "input"
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # Get all .txt files in the input directory
    tenders = [f for f in os.listdir(input_dir) if f.endswith('.txt')]

    if not tenders:
        print(f"No tender documents found in {input_dir}")
        return

    print(f"Found {len(tenders)} tenders to process.")

    for filename in tenders:
        input_path = os.path.join(input_dir, filename)
        output_filename = os.path.splitext(filename)[0] + ".json"
        output_path = os.path.join(output_dir, output_filename)

        print(f"\n--- Starting Extraction for: {filename} ---")
        print(f"Input: {input_path}")
        print(f"Output: {output_path}")

        # Start the workflow
        # We use a unique workflow ID based on the filename to avoid collisions
        workflow_id = f"extract-{os.path.splitext(filename)[0]}"
        
        try:
            handle = await client.start_workflow(
                TenderExtractionWorkflow.run,
                args=[input_path, output_path],
                id=workflow_id,
                task_queue="tender-extraction-queue",
            )

            print(f"Workflow started. ID: {handle.id}, Run ID: {handle.result_run_id}")
            # In a real production scenario, you might want to wait or just fire-and-forget
            # result = await handle.result()
            # print(f"Result saved to: {result}")
        except Exception as e:
            print(f"Error starting workflow for {filename}: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
