from typing import Any

import mlflow
import pandas as pd
from feast import FeatureStore

from src.config import (
  FEATURE_STORE_REPO,
  MLFLOW_TRACKING_URI,
  PARQUET_PATH,
  PREDICT_MODEL_ALIAS,
  PREDICT_MODEL_NAME,
  PREDICT_MODEL_VERSION,
)

ONLINE_FEATURES = [
  "well_stats:tipoextraccion",
  "well_stats:avg_prod_gas_10m",
  "well_stats:avg_prod_pet_10m",
  "well_stats:last_prod_gas",
  "well_stats:last_prod_pet",
  "well_stats:n_readings",
]
MODEL_FEATURES = [
  "tipoextraccion",
  "avg_prod_gas_10m",
  "avg_prod_pet_10m",
  "last_prod_gas",
  "last_prod_pet",
  "n_readings",
]


def _load_model(target: str) -> Any:
  mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
  model_name = f"{target}{PREDICT_MODEL_NAME}"

  if PREDICT_MODEL_VERSION is not None:
    model_uri = f"models:/{model_name}/{PREDICT_MODEL_VERSION}"
  else:
    model_uri = f"models:/{model_name}@{PREDICT_MODEL_ALIAS}"

  return mlflow.sklearn.load_model(model_uri)


def _build_model_input(
    tipoextraccion: Any,
    avg_prod_gas_10m: Any,
    avg_prod_pet_10m: Any,
    last_prod_gas: Any,
    last_prod_pet: Any,
    n_readings: Any,
    model: Any,
) -> pd.DataFrame:
  row = {
    "tipoextraccion": tipoextraccion,
    "avg_prod_gas_10m": float(avg_prod_gas_10m) if pd.notna(avg_prod_gas_10m) else 0.0,
    "avg_prod_pet_10m": float(avg_prod_pet_10m) if pd.notna(avg_prod_pet_10m) else 0.0,
    "last_prod_gas": float(last_prod_gas) if pd.notna(last_prod_gas) else 0.0,
    "last_prod_pet": float(last_prod_pet) if pd.notna(last_prod_pet) else 0.0,
    "n_readings": int(max(1, min(10, int(n_readings) if pd.notna(n_readings) else 10))),
  }

  X_df = pd.DataFrame([row], columns=MODEL_FEATURES)
  X_df = pd.get_dummies(X_df, columns=["tipoextraccion"], drop_first=False)
  X_df = X_df.fillna(0)

  if hasattr(model, "feature_names_in_"):
    expected_cols = list(model.feature_names_in_)
    X_df = X_df.reindex(columns=expected_cols, fill_value=0)

  return X_df


def _predict_from_context_row(row: pd.Series, model: Any) -> float:
  X_df = _build_model_input(
    tipoextraccion=row.get("tipoextraccion"),
    avg_prod_gas_10m=row.get("avg_prod_gas_10m"),
    avg_prod_pet_10m=row.get("avg_prod_pet_10m"),
    last_prod_gas=row.get("last_prod_gas"),
    last_prod_pet=row.get("last_prod_pet"),
    n_readings=row.get("n_readings"),
    model=model,
  )
  pred = float(model.predict(X_df)[0])
  return max(0.0, pred)


def _get_latest_offline_date_for_well(id_well: int) -> pd.Timestamp | None:
  offline_df = pd.read_parquet(PARQUET_PATH, columns=["idpozo", "fecha"])
  offline_df = offline_df[offline_df["idpozo"] == int(id_well)].copy()
  if offline_df.empty:
    return None

  offline_df["fecha"] = pd.to_datetime(offline_df["fecha"])
  return offline_df["fecha"].max().to_period("M").to_timestamp()


def _get_online_features_if_available(
    store: FeatureStore,
    id_well: int,
    requested_date: pd.Timestamp,
) -> pd.Series | None:
  latest_offline_date = _get_latest_offline_date_for_well(id_well=id_well)
  if latest_offline_date is None or requested_date != latest_offline_date:
    return None

  online_features = store.get_online_features(
    features=ONLINE_FEATURES,
    entity_rows=[{"idpozo": int(id_well)}],
  ).to_dict()

  if not online_features:
    return None

  row = {}
  for feature_name in ONLINE_FEATURES:
    short_name = feature_name.split(":")[-1]
    values = online_features.get(short_name)
    if not values:
      return None

    value = values[0]
    if pd.isna(value):
      return None

    row[short_name] = value

  return pd.Series(row)


def _load_offline_rows_for_range(id_well: int, forecast_dates: pd.DatetimeIndex) -> pd.DataFrame:
  offline_df = pd.read_parquet(PARQUET_PATH)
  offline_df["fecha"] = pd.to_datetime(offline_df["fecha"]).dt.to_period("M").dt.to_timestamp()

  rows = offline_df[
    (offline_df["idpozo"] == int(id_well)) &
    (offline_df["fecha"].isin(forecast_dates))
  ].copy()

  requested_set = set(forecast_dates)
  found_set = set(rows["fecha"])
  missing = sorted(requested_set - found_set)
  if missing:
    missing_dates = ", ".join(d.strftime("%Y-%m-%d") for d in missing)
    raise ValueError(
      "No hay contexto suficiente en offline store para todas las fechas solicitadas. "
      f"Fechas faltantes para id_well={id_well}: {missing_dates}"
    )

  return rows.sort_values("fecha").reset_index(drop=True)


def predict(
    target: str,
    id_well: int,
    date_start: str,
    date_end: str,
):
  start = pd.Timestamp(date_start).to_period("M").to_timestamp()
  end = pd.Timestamp(date_end).to_period("M").to_timestamp()
  if start > end:
    raise ValueError("date_start debe ser menor o igual a date_end.")

  forecast_dates = pd.date_range(start=start, end=end, freq="MS")
  if forecast_dates.empty:
    return {"id_well": int(id_well), "forecast": []}

  model = _load_model(target=target)
  store = FeatureStore(repo_path=FEATURE_STORE_REPO)

  forecast = []
  online_row = None
  if len(forecast_dates) == 1:
    online_row = _get_online_features_if_available(
      store=store,
      id_well=id_well,
      requested_date=forecast_dates[0],
    )

  if online_row is not None:
    print(
      f"Using online feature store for id_well={id_well}, date={forecast_dates[0].strftime('%Y-%m-%d')}"
    )
    row = online_row
    pred = _predict_from_context_row(row=row, model=model)
    forecast.append({"date": forecast_dates[0].strftime("%Y-%m-%d"), "prod": pred})
  else:
    print(
      "Using offline feature store for "
      f"id_well={id_well}, range={forecast_dates[0].strftime('%Y-%m-%d')}"
      f" to {forecast_dates[-1].strftime('%Y-%m-%d')}"
    )
    offline_rows = _load_offline_rows_for_range(id_well=id_well, forecast_dates=forecast_dates)
    for _, row in offline_rows.iterrows():
      pred = _predict_from_context_row(row=row, model=model)
      forecast.append({"date": row["fecha"].strftime("%Y-%m-%d"), "prod": pred})

  return {
    "id_well": int(id_well),
    "forecast": forecast,
  }


if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument("--target", type=str, default="prod_gas")
  parser.add_argument("--id_well", type=int, required=True)
  parser.add_argument("--date_start", type=str, required=True)
  parser.add_argument("--date_end", type=str, required=True)
  args = parser.parse_args()

  result = predict(
    target=args.target,
    id_well=args.id_well,
    date_start=args.date_start,
    date_end=args.date_end,
  )
  print(result)