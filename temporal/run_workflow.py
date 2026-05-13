import asyncio
import os
import sys
from temporalio.client import Client
from temporal.workflows import TenderExtractionWorkflow

async def main():
    # Connect to Temporal
    temporal_url = os.getenv("TEMPORAL_URL", "localhost:7233")
    client = await Client.connect(temporal_url)

    input_dir = "input"
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # Get all .txt files in the input directory for listing
    available_tenders = [f for f in os.listdir(input_dir) if f.endswith('.txt')]

    # Check for command line argument
    if len(sys.argv) < 2:
        print("\n❌ Error: No tender filename provided.")
        print("\nUsage: python -m temporal.run_workflow <filename.txt>")
        if available_tenders:
            print("\nAvailable tenders in 'input/':")
            for t in available_tenders:
                print(f"  - {t}")
        return

    target_file = sys.argv[1]
    input_path = os.path.join(input_dir, target_file)

    if not os.path.exists(input_path):
        print(f"❌ Error: File not found: {input_path}")
        return

    output_filename = os.path.splitext(target_file)[0] + ".json"
    output_path = os.path.join(output_dir, output_filename)

    print(f"\n🚀 Triggering Manual Extraction for: {target_file}")
    print(f"Output will be saved to: {output_path}")

    # Start the workflow
    workflow_id = f"extract-{os.path.splitext(target_file)[0].replace(' ', '_')}"
    
    try:
        handle = await client.start_workflow(
            TenderExtractionWorkflow.run,
            args=[input_path, output_path],
            id=workflow_id,
            task_queue="tender-extraction-queue",
        )

        print(f"✅ Workflow started successfully!")
        print(f"Workflow ID: {handle.id}")
        print(f"View status at: http://localhost:8080/namespaces/default/workflows/{handle.id}")
        
    except Exception as e:
        print(f"❌ Error starting workflow: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
