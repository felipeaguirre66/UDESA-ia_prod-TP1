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
