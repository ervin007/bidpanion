from datetime import timedelta
from temporalio import workflow
from config import FIELDS
import logging

# Import activities (only for type hinting in Workflow.execute)
with workflow.unsafe.imports_passed_through():
    from temporal.activities import ExtractionActivities

from temporalio.common import RetryPolicy

@workflow.defn
class FieldExtractionWorkflow:
    @workflow.run
    async def run(self, field_id: str, input_file: str) -> dict:
        return await workflow.execute_activity_method(
            ExtractionActivities.extract_field_activity,
            args=[field_id, input_file],
            start_to_close_timeout=timedelta(minutes=15),
            retry_policy=RetryPolicy(maximum_attempts=1)
        )

@workflow.defn
class TenderExtractionWorkflow:
    @workflow.run
    async def run(self, input_file: str, output_file: str) -> dict:
        # 1. Ingest and prepare indices
        await workflow.execute_activity_method(
            ExtractionActivities.prepare_indices,
            args=[input_file],
            start_to_close_timeout=timedelta(minutes=10),
        )

        # 2. Extract each field in batches to avoid overwhelming Vertex AI quota
        # 2. Extract each field (Full Parallel)
        import asyncio
        extraction_futures = []
        for field in FIELDS:
            extraction_futures.append(
                workflow.execute_child_workflow(
                    FieldExtractionWorkflow,
                    args=[field["id"], input_file],
                    id=f"{workflow.info().workflow_id}-field-{field['id']}",
                    retry_policy=RetryPolicy(maximum_attempts=1),
                    execution_timeout=timedelta(minutes=20)
                )
            )
        
        results = await asyncio.gather(*extraction_futures)




        # 3. Finalize and save (using a helper to structure the JSON)
        # Note: aggregation logic could also be done here in the workflow 
        # but file writing must be an activity.
        
        # Build the structured output (logic from main.py)
        # For simplicity in this example, let's assume we have a SaveActivity
        # that handles the final formatting and writing.
        
        await workflow.execute_activity_method(
            ExtractionActivities.save_final_results,
            args=[results, output_file],
            start_to_close_timeout=timedelta(minutes=2),
        )

        return {"status": "completed", "output": output_file}
