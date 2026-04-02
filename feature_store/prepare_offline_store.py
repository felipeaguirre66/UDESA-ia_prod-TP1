import urllib.request
import pandas as pd
from src.config import RAW_DATASET_PATH, PARQUET_PATH

DATASET_DOWNLOAD_URL = "http://datos.energia.gob.ar/dataset/c846e79c-026c-4040-897f-1ad3543b407c/resource/b5b58cdc-9e07-41f9-b392-fb9ec68b0725/download/produccin-de-pozos-de-gas-y-petrleo-no-convencional.csv"

def download_data():
  print("Descargando dataset...")
  urllib.request.urlretrieve(DATASET_DOWNLOAD_URL, str(RAW_DATASET_PATH))
  print("Descarga completada.")

def prepare_offline_store():
    print("Preparando features históricas para el offline store...")
    df = pd.read_csv(RAW_DATASET_PATH)
    df['fecha'] = pd.to_datetime(
        df['anio'].astype(str) + '-' + df['mes'].astype(str).str.zfill(2) + '-01'
    )
    df = df[['idpozo', 'fecha', 'prod_pet', 'prod_gas', 'prod_agua', 'tef', 'profundidad', 'tipoextraccion']].dropna()
    df = df.sort_values(['idpozo', 'fecha']).reset_index(drop=True)
    records = []

    for well_id, group in df.groupby('idpozo', sort=False):
        group = group.sort_values('fecha').reset_index(drop=True)
        records_before_group = len(records)

        for i in range(10, len(group)):
            window  = group.iloc[i - 10: i]
            current = group.iloc[i].to_dict()

            current['avg_prod_gas_10m'] = float(window['prod_gas'].mean())
            current['avg_prod_pet_10m'] = float(window['prod_pet'].mean())
            current['last_prod_gas']    = float(window['prod_gas'].iloc[-1])
            current['last_prod_pet']    = float(window['prod_pet'].iloc[-1])
            current['n_readings']       = int(len(window))
            records.append(current)

        # Overwrite the last record with the future prediction row when this group produced rows.
        if len(records) > records_before_group:
            records[-1]['fecha']    = records[-1]['fecha'] + pd.DateOffset(months=1)
            records[-1]['prod_gas'] = None
            records[-1]['prod_pet'] = None
        
    feat_df = pd.DataFrame(records)
    #TODO: remove this, use only for memory issues
    feat_df = feat_df.sort_values('fecha').groupby('idpozo').tail(5).reset_index(drop=True)
    feat_df.to_parquet(PARQUET_PATH, index=False)
    print(f"Features offline guardadas en {PARQUET_PATH} con {len(feat_df)} filas.")

if __name__ == "__main__":
  download_data()
  prepare_offline_store()