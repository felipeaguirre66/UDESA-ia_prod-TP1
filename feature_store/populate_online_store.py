import os
import subprocess
import pandas as pd
from src.config import FEATURE_STORE_REPO, PARQUET_PATH

def populate_online_store():
  print("Materializando features al Online Store...")
  from feast import FeatureStore

  feat_df = pd.read_parquet(PARQUET_PATH)
  latest_df = feat_df.sort_values('fecha').groupby('idpozo').tail(1)
  
  store = FeatureStore(repo_path=str(FEATURE_STORE_REPO))
  store.write_to_online_store(
    feature_view_name="well_stats",
    df=latest_df,
  )
  print("Materialización exitosa.")
  
  # Demostración
  print("\n[Online Store] Validando lectura...")
  pozo_ejemplo = 132879
  
  features = store.get_online_features(
    features=[
      "well_stats:avg_prod_gas_10m",
      "well_stats:n_readings"
    ],
    entity_rows=[{"idpozo": pozo_ejemplo}]
  ).to_dict()
  
  print(f"Lectura exitosa. Features del pozo {pozo_ejemplo}:")
  for key, value in features.items():
    print(f"  {key}: {value[0]}")

def apply_feast():
    result = subprocess.run(["feast", "apply"], capture_output=True, text=True, cwd=str(FEATURE_STORE_REPO))
    if result.returncode != 0:
        print("Error applying Feast:")
        print(result.stderr)
        raise Exception("Feast apply failed")
    else:
        print("Feast applied successfully:")
        print(result.stdout)

if __name__ == "__main__":
  populate_online_store()
