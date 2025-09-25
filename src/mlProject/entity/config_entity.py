from dataclasses import dataclass
from pathlib import Path


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