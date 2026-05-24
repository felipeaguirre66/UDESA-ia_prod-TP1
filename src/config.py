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

# Shared feature definitions used across training, inference and monitoring.
CATEGORICAL_FEATURE = "tipoextraccion"
NUMERICAL_FEATURES = [
	"avg_prod_gas_10m",
	"avg_prod_pet_10m",
	"last_prod_gas",
	"last_prod_pet",
	"n_readings",
]
MODEL_FEATURES = [CATEGORICAL_FEATURE, *NUMERICAL_FEATURES]

# Drift monitoring defaults.
DRIFT_RECENT_MONTHS = 3
DRIFT_SIGNIFICANCE_LEVEL = 0.05