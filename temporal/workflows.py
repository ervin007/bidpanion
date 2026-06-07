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
    async def run(self, input_file: str, output_file: str, callback_url: str = None, company_profile: str = None) -> dict:
        try:
            # 1. Unzip and parse tender zip
            parsed_txt_file = await workflow.execute_activity_method(
                ExtractionActivities.parse_zip_file_activity,
                args=[input_file],
                start_to_close_timeout=timedelta(minutes=15),
            )

            # 2. Ingest and prepare indices
            await workflow.execute_activity_method(
                ExtractionActivities.prepare_indices,
                args=[parsed_txt_file],
                start_to_close_timeout=timedelta(minutes=10),
            )

            # 3. Extract each field (Full Parallel)
            import asyncio
            extraction_futures = []
            for field in FIELDS:
                extraction_futures.append(
                    workflow.execute_child_workflow(
                        FieldExtractionWorkflow,
                        args=[field["id"], parsed_txt_file],
                        id=f"{workflow.info().workflow_id}-field-{field['id']}",
                        retry_policy=RetryPolicy(maximum_attempts=1),
                        execution_timeout=timedelta(minutes=20)
                    )
                )
            
            results = await asyncio.gather(*extraction_futures)

            # 4. Finalize and save
            await workflow.execute_activity_method(
                ExtractionActivities.save_final_results,
                args=[results, output_file],
                start_to_close_timeout=timedelta(minutes=2),
            )

            # 5. Calculate Fit Score if company profile is provided
            fit_results = None
            if company_profile:
                fit_results = await workflow.execute_activity_method(
                    ExtractionActivities.calculate_fit_score,
                    args=[results, company_profile],
                    start_to_close_timeout=timedelta(minutes=10),
                )

            if callback_url:
                await workflow.execute_activity_method(
                    ExtractionActivities.send_completion_webhook_activity,
                    args=[callback_url, workflow.info().workflow_id, input_file, "completed", output_file, fit_results],
                    start_to_close_timeout=timedelta(minutes=5),
                    retry_policy=RetryPolicy(
                        maximum_attempts=10, 
                        initial_interval=timedelta(seconds=5), 
                        backoff_coefficient=2.0
                    )
                )

            return {"status": "completed", "output": output_file}

        except Exception as e:
            if callback_url:
                await workflow.execute_activity_method(
                    ExtractionActivities.send_completion_webhook_activity,
                    args=[callback_url, workflow.info().workflow_id, input_file, "failed", None, None],
                    start_to_close_timeout=timedelta(minutes=5),
                    retry_policy=RetryPolicy(
                        maximum_attempts=10, 
                        initial_interval=timedelta(seconds=5), 
                        backoff_coefficient=2.0
                    )
                )
            raise e
