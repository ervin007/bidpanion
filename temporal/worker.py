import asyncio
import os
from temporalio.client import Client
from temporalio.worker import Worker
from temporal.activities import ExtractionActivities
from temporal.workflows import TenderExtractionWorkflow, FieldExtractionWorkflow

async def main():
    # Connect to Temporal server
    temporal_url = os.environ.get("TEMPORAL_URL", "localhost:7233")
    client = await Client.connect(temporal_url)

    # Initialize activities
    activities = ExtractionActivities()

    # Create worker
    worker = Worker(
        client,
        task_queue="tender-extraction-queue",
        workflows=[TenderExtractionWorkflow, FieldExtractionWorkflow],
        activities=[
            activities.prepare_indices,
            activities.extract_field_activity,
            activities.save_final_results,
            activities.ingest_document,
        ],
    )

    print(f"Worker started on queue 'tender-extraction-queue' connecting to {temporal_url}")
    await worker.run()

if __name__ == "__main__":
    asyncio.run(main())
