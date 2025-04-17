from hr_attrition.config.configuration import ConfigurationManager
from hr_attrition.components.model_evaluation import ModelEvaluation  # fixed import
from hr_attrition.utils.logging import logger  # fixed logger import
 # fixed import




STAGE_NAME = "Model Evaluation stage"

class ModelEvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config = ConfigurationManager()

    # Get the model evaluation configuration
            model_eval_config = config.get_model_evaluation_config()

    # Initialize the ModelEvaluation class
            model_evaluation = ModelEvaluation(config=model_eval_config)

    # Perform evaluation and save results
            model_evaluation.save_results()
        except FileNotFoundError as e:
            print(f"File not found: {e}")
            raise e
        except Exception as e:
            logger.error(f"Exception occurred in {STAGE_NAME}: {e}")
            raise e
       
