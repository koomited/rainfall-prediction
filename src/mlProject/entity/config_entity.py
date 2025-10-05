from pathlib import Path
from dataclasses import dataclass


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path


@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    data_dir: Path
    STATUS_FILE: str
    all_schema: str


@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    data_path: Path
    all_schema: str
    test_size: float
    look_back: int


@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    train_data_path: Path
    test_data_path: Path
    scaler_name: str
    classification_model_name: str
    lstm_model_name: str
    classification_model_params: dict
    lstm_model_params: dict
    target_column: dict
    target_column_classification: dict
    batch_size: int


@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    test_data_path: Path
    scaler_path: Path
    classifier_path: Path
    regressor_path: Path
    classifier_params: dict
    regressor_params: dict
    metric_file_name: str
    classification_target_column: str
    regression_target_column: str
    mlflow_uri: str
    batch_size: int
