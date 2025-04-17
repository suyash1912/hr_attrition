from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path



from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    STATUS_FILE: str
    unzip_data_dir: Path
    all_schema: dict



from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataTransformationConfig:
    data_path: Path
    root_dir: Path
    test_size: float
    random_state: int




from dataclasses import dataclass
from pathlib import Path

@dataclass
class ModelTrainingConfig:
    root_dir: Path
    train_data_path: Path
    test_data_path: Path
    model_path: Path
    C: float
    penalty: str
    solver: str
    class_weight: str  # use Optional[str] if you plan to set None
    max_iter: int
    random_state: int






@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    test_data_path: Path
    model_path: Path
    all_params: dict
    metric_file_name: Path
    target_column: str

   