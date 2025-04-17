from hr_attrition.utils.logging import logger
from hr_attrition.pipeline.data_ingestion import DataIngestionTrainingPipeline
from hr_attrition.pipeline.data_validation import DataValidationTrainingPipeline
from hr_attrition.pipeline.data_transformation import DataTransformationTrainingPipeline
from hr_attrition.pipeline.model_trainer import ModelTrainingPipeline
from hr_attrition.pipeline.model_evaluation import ModelEvaluationPipeline  # Fixed import


def run_pipeline(stage_name, pipeline_class):
    """
    Helper function to run a pipeline stage with proper logging and error handling.
    """
    try:
        logger.info(f">>>>>> stage {stage_name} started <<<<<<")
        pipeline = pipeline_class()
        pipeline.main()
        logger.info(f">>>>>> stage {stage_name} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.error(f"Exception occurred in stage {stage_name}: {e}")
        logger.exception(e)
        raise e


if __name__ == "__main__":
    # Define the stages of the pipeline
    stages = [
        ("Data Ingestion stage", DataIngestionTrainingPipeline),
        ("Data Validation stage", DataValidationTrainingPipeline),
        ("Data Transformation stage", DataTransformationTrainingPipeline),
        ("Model Training stage", ModelTrainingPipeline),
        ("Model Evaluation stage", ModelEvaluationPipeline),  # Fixed pipeline reference
    ]

    # Run each stage sequentially
    for stage_name, pipeline_class in stages:
        run_pipeline(stage_name, pipeline_class)

