from hr_attrition.config.configuration import ConfigurationManager
from hr_attrition.components.model_trainer import ModelTrainer  # fixed import
from hr_attrition.utils.logging import logger  # fixed logger import

STAGE_NAME = "Model Training stage"

class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            model_trainer_config = config.get_model_trainer_config()
            
            model_trainer = ModelTrainer(config=model_trainer_config)
            model_trainer.train()

        except Exception as e:
            logger.error(f"Exception occurred in {STAGE_NAME}: {e}")
            raise e
