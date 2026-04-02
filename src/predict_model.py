import os
import pickle
import pandas as pd
from feast import FeatureStore

FEATURE_STORE_REPO = "./feature_store"
MODELS_DIR = "./models"


def _get_latest_model_path(models_dir: str) -> str:
  if not os.path.isdir(models_dir):
    raise FileNotFoundError(f"No existe el directorio de modelos: {models_dir}")

  model_files = [
    f for f in os.listdir(models_dir)
    if f.endswith(".pkl") and os.path.isfile(os.path.join(models_dir, f))
  ]
  if not model_files:
    raise FileNotFoundError(f"No se encontraron archivos .pkl en: {models_dir}")

  # Ordenamos por prefijo fecha (YYYY-MM-DD) y, como desempate, por mtime.
  def _sort_key(filename: str):
    date_part = filename.split("__", 1)[0]
    parsed = pd.to_datetime(date_part, errors="coerce")
    ts = parsed.value if pd.notna(parsed) else -1
    mtime = os.path.getmtime(os.path.join(models_dir, filename))
    return (ts, mtime)

  latest_file = max(model_files, key=_sort_key)
  return os.path.join(models_dir, latest_file)

def predict():

  store = FeatureStore(repo_path=FEATURE_STORE_REPO)
  idpozo_ejemplo = 132879

  latest_model_path = _get_latest_model_path(MODELS_DIR)
  print(f"Cargando modelo mas reciente: {latest_model_path}")
  with open(latest_model_path, "rb") as f:
    model = pickle.load(f)

  print(f"Consultando contexto online para el pozo {idpozo_ejemplo}...")
  online_features = store.get_online_features(
    features=[
      "well_stats:tipoextraccion",
      "well_stats:avg_prod_gas_10m",
      "well_stats:avg_prod_pet_10m",
      "well_stats:last_prod_gas",
      "well_stats:last_prod_pet",
      "well_stats:n_readings",
    ],
    entity_rows=[{"idpozo": idpozo_ejemplo}],
  ).to_df()

  X_df = online_features[
    [
      "tipoextraccion",
      "avg_prod_gas_10m",
      "avg_prod_pet_10m",
      "last_prod_gas",
      "last_prod_pet",
      "n_readings",
    ]
  ].copy()

  # Aplicamos el mismo encoding usado durante entrenamiento.
  X_df = pd.get_dummies(X_df, columns=["tipoextraccion"], drop_first=False)
  X_df = X_df.fillna(0)

  # Alineamos columnas para evitar mismatch entre train/predict.
  if hasattr(model, "feature_names_in_"):
    expected_cols = list(model.feature_names_in_)
    X_df = X_df.reindex(columns=expected_cols, fill_value=0)

  pred = float(model.predict(X_df)[0])
  print(f"Prediccion prod_gas para idpozo={idpozo_ejemplo}: {pred:.4f}")
  return {
    "idpozo": idpozo_ejemplo,
    "model_path": latest_model_path,
    "prediction": pred,
  }


if __name__ == "__main__":
  predict()