from hr_attrition.constants import *
from hr_attrition.utils.common import read_yaml, create_directories
from hr_attrition.entity.config_entity import DataIngestionConfig, DataValidationConfig, DataTransformationConfig , ModelTrainingConfig , ModelEvaluationConfig


class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH,
        schema_filepath = SCHEMA_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)

        create_directories([self.config.artifacts_root])


    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config
    

    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation
        schema = self.schema.COLUMNS

        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
            root_dir=config.root_dir,
            STATUS_FILE=config.STATUS_FILE,
            unzip_data_dir = config.unzip_data_dir,
            all_schema=schema,
        )

        return data_validation_config
    



    def get_data_transformation_config(self) -> DataTransformationConfig:
     config = self.config.data_transformation
     params = self.params.data_transformation

     create_directories([config.root_dir])

     data_transformation_config = DataTransformationConfig(
        root_dir=Path(config.root_dir),
        data_path=Path(config.data_path),
        test_size=params.test_size,
        random_state=params.random_state
    )

     return data_transformation_config
    

    
    def get_model_trainer_config(self) -> ModelTrainingConfig:
     config = self.config.model_training
     params = self.params.model_params

     create_directories([config.root_dir])

     model_trainer_config = ModelTrainingConfig(
        root_dir=Path(config.root_dir),
        train_data_path=Path(config.train_data_path),
        test_data_path=Path(config.test_data_path),
        model_path=Path(config.model_path),
        C=params.C,
        penalty=params.penalty,
        solver=params.solver,
        class_weight=params.class_weight,
        max_iter=params.max_iter,
        random_state=params.random_state
    )

     return model_trainer_config
    

   



    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation
        params = self.params.model_params
        schema = self.schema.TARGET_COLUMN

        # Ensure the root directory for model evaluation exists
        create_directories([config.root_dir])

        # Create and return the ModelEvaluationConfig object
        model_evaluation_config = ModelEvaluationConfig(
            root_dir=Path(config.root_dir),
            test_data_path=Path(config.test_data_path),
            model_path=Path(config.model_path),
            all_params=params,
            metric_file_name=Path(config.metric_file_name),
            target_column=schema.name
        )
        return model_evaluation_config
        



