from pathlib import Path

# Docker container paths (as mounted in docker-compose.yaml)
FEATURE_STORE_REPO = Path("/opt/airflow/feature_store")
DATA_DIR = Path("/opt/airflow/data")
RAW_DATASET_PATH = DATA_DIR / "dataset.csv"
PARQUET_PATH = FEATURE_STORE_REPO / "data" / "well_features.parquet"
MODELS_DIR = Path("/opt/airflow/model")