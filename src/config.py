from pathlib import Path

# Docker container paths (as mounted in docker-compose.yaml)
FEATURE_STORE_REPO = Path("/opt/airflow/feature_store")
DATA_DIR = Path("/opt/airflow/data")
RAW_DATASET_PATH = DATA_DIR / "dataset.csv"
PARQUET_PATH = FEATURE_STORE_REPO / "data" / "well_features.parquet"

# MLflow registry configuration used at inference time.
MLFLOW_TRACKING_URI = "http://mlflow:9090"

# Global model selector for predict().
# - If PREDICT_MODEL_VERSION is set, that version is loaded.
# - Otherwise, alias PREDICT_MODEL_ALIAS is loaded.
PREDICT_MODEL_NAME = "__random_forest"
PREDICT_MODEL_ALIAS = "champion"
PREDICT_MODEL_VERSION = None