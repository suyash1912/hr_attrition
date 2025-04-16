from hr_attrition.config.configuration import ConfigurationManager
from hr_attrition.components.data_validation import DataValidation
from hr_attrition.utils.logging import logger


STAGE_NAME = "Data Validation stage"

class DataValidationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_validation_config = config.get_data_validation_config()
        data_validation = DataValidation(config=data_validation_config)
        data_validation.validate_all_columns()