from mlProject.constants import *
from mlProject.utils.common import read_yaml, create_directories
from mlProject.entity.config_entity import (
    ModelTrainerConfig,
    DataIngestionConfig,
    DataValidationConfig,
    ModelEvaluationConfig,
    DataTransformationConfig,
)


class ConfigurationManager:
    def __init__(
        self,
        config_filepath: str = CONFIG_FILE_PATH,
        params_filepath: str = PARAMS_FILE_PATH,
        schema_filepath: str = SCHEMA_FILE_PATH,
    ):
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
        )
        return data_ingestion_config

    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation
        schema = self.schema

        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
            root_dir=config.root_dir,
            STATUS_FILE=config.STATUS_FILE,
            data_dir=config.data_dir,
            all_schema=schema,
        )

        return data_validation_config

    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation
        schema = self.schema
        create_directories([config.root_dir])
        data_transformation_config = DataTransformationConfig(
            root_dir=Path(config.root_dir),
            data_path=Path(config.data_path),
            all_schema=schema,
            test_size=config.test_size,
            look_back=config.look_back,
        )

        return data_transformation_config

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        models_params = self.params
        schema = self.schema

        create_directories([config.root_dir])

        model_trainer_config = ModelTrainerConfig(
            root_dir=config.root_dir,
            train_data_path=config.train_data_path,
            test_data_path=config.test_data_path,
            scaler_name=config.scaler_name,
            classification_model_name=config.classification_model_name,
            lstm_model_name=config.lstm_model_name,
            classification_model_params=models_params.classification,
            lstm_model_params=models_params.regression,
            target_column=schema.TARGET_COLUMN,
            target_column_classification=schema.TARGET_COLUMN_CLASSIFICATION,
            batch_size=config.batch_size,
        )

        return model_trainer_config

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation
        params = self.params
        schema = self.schema

        create_directories([config.root_dir])

        model_evaluation_config = ModelEvaluationConfig(
            root_dir=config.root_dir,
            test_data_path=config.test_data_path,
            scaler_path=config.scaler_path,
            classifier_path=config.classifier_path,
            regressor_path=config.regressor_path,
            classifier_params=params.classification,
            regressor_params=params.regression,
            metric_file_name=config.metric_file_name,
            classification_target_column=schema.TARGET_COLUMN_CLASSIFICATION,
            regression_target_column=schema.TARGET_COLUMN,
            mlflow_uri=config.mlflow_uri,
            batch_size=config.batch_size,
        )

        return model_evaluation_config
