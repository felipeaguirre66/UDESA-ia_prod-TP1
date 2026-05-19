from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from tqdm.auto import tqdm

from src.config import CATEGORICAL_FEATURE, MODEL_FEATURES, RAW_DATASET_PATH

MODEL_PARAMS = {"n_estimators": 200, "random_state": 42, "n_jobs": -1}
WINDOW_MONTHS = 2
MIN_ROWS = 100
DEFAULT_TARGET = "prod_gas"
MIN_ANALYSIS_DATE = pd.Timestamp("2025-01-01")
MAE_THRESHOLD = 155


def _load_dataset(target: str) -> pd.DataFrame:
  """Load the raw monthly dataset and build leakage-free rolling features.

  The feature definitions mirror the ones used in training:
  - each row is predicted using the previous 10 monthly readings of the same well
  - features are computed only from past rows (shifted by one)
  """
  df = pd.read_csv(RAW_DATASET_PATH)
  df["fecha"] = pd.to_datetime(
    df["anio"].astype(str) + "-" + df["mes"].astype(str).str.zfill(2) + "-01"
  )

  needed_cols = []
  for col in ["idpozo", "fecha", target, "prod_gas", "prod_pet", "tipoextraccion"]:
    if col not in needed_cols:
      needed_cols.append(col)
  df = df[needed_cols].dropna(subset=["idpozo", "fecha", target, "prod_gas", "prod_pet", "tipoextraccion"])
  df = df.sort_values(["idpozo", "fecha"]).reset_index(drop=True)

  def _add_rolling_features(group: pd.DataFrame) -> pd.DataFrame:
    group = group.sort_values("fecha").reset_index(drop=True).copy()
    past_gas = group["prod_gas"].shift(1)
    past_pet = group["prod_pet"].shift(1)
    group["avg_prod_gas_10m"] = past_gas.rolling(window=10, min_periods=10).mean()
    group["avg_prod_pet_10m"] = past_pet.rolling(window=10, min_periods=10).mean()
    group["last_prod_gas"] = past_gas
    group["last_prod_pet"] = past_pet
    group["n_readings"] = past_gas.rolling(window=10, min_periods=10).count()
    return group

  parts: list[pd.DataFrame] = []
  for well_id, group in df.groupby("idpozo", sort=False):
    g = group.copy()
    g["idpozo"] = well_id
    parts.append(_add_rolling_features(g))

  feature_df = pd.concat(parts, ignore_index=True)

  feature_df = feature_df.dropna(subset=["avg_prod_gas_10m", "avg_prod_pet_10m", "last_prod_gas", "last_prod_pet", "n_readings", target])
  return feature_df[["idpozo", "fecha", target, *MODEL_FEATURES]].reset_index(drop=True)


def _fit_train_and_score(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target: str,
) -> tuple[float, int, int]:
  feature_cols = list(MODEL_FEATURES)

  X_train = pd.get_dummies(train_df[feature_cols], columns=[CATEGORICAL_FEATURE], drop_first=False)
  y_train = train_df[target].astype(float)

  X_test = pd.get_dummies(test_df[feature_cols], columns=[CATEGORICAL_FEATURE], drop_first=False)
  X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
  y_test = test_df[target].astype(float)

  model = RandomForestRegressor(**MODEL_PARAMS)
  model.fit(X_train, y_train)
  preds = model.predict(X_test)
  mae = float(mean_absolute_error(y_test, preds))
  return mae, len(train_df), len(test_df)


def _month_add(ts: pd.Timestamp, months: int) -> pd.Timestamp:
  return (ts + pd.DateOffset(months=months)).to_period("M").to_timestamp()


def _build_windows(df: pd.DataFrame, window_months: int = WINDOW_MONTHS) -> list[dict]:
  min_month = max(df["fecha"].min().to_period("M").to_timestamp(), MIN_ANALYSIS_DATE.to_period("M").to_timestamp())
  max_month = df["fecha"].max().to_period("M").to_timestamp()

  windows: list[dict] = []
  window_end = max_month
  window_index = 1

  while window_end >= min_month:
    window_start = _month_add(window_end, -(window_months - 1))
    test_mask = (df["fecha"] >= window_start) & (df["fecha"] <= window_end)
    train_mask = df["fecha"] < window_start

    train_df = df.loc[train_mask].copy()
    test_df = df.loc[test_mask].copy()

    windows.append(
      {
        "window_index": window_index,
        "train_df": train_df,
        "test_df": test_df,
        "train_date_start": train_df["fecha"].min() if not train_df.empty else pd.NaT,
        "train_date_end": train_df["fecha"].max() if not train_df.empty else pd.NaT,
        "test_date_start": test_df["fecha"].min() if not test_df.empty else pd.NaT,
        "test_date_end": test_df["fecha"].max() if not test_df.empty else pd.NaT,
      }
    )

    window_index += 1
    window_end = _month_add(window_start, -1)

  return windows


def run_model_decay(target: str = DEFAULT_TARGET, window_months: int = WINDOW_MONTHS) -> dict:
  print("[model_decay] Loading dataset and building leakage-safe features...", flush=True)
  df = _load_dataset(target=target)
  print(f"[model_decay] Dataset ready: rows={len(df)}, min_date={df['fecha'].min().date()}, max_date={df['fecha'].max().date()}", flush=True)

  print("[model_decay] Building time windows...", flush=True)
  windows = _build_windows(df, window_months=window_months)
  print(f"[model_decay] Total windows to evaluate: {len(windows)}", flush=True)

  results: list[dict] = []
  skipped_windows: list[dict] = []

  for window in tqdm(windows, desc="model_decay_windows", unit="window"):
    train_df = window["train_df"]
    test_df = window["test_df"]
    idx = window["window_index"]
    print(
      f"[model_decay] Window {idx}/{len(windows)} | "
      f"train_rows={len(train_df)} test_rows={len(test_df)} | "
      f"test_range={window['test_date_start']}..{window['test_date_end']}",
      flush=True,
    )

    if len(train_df) < MIN_ROWS or len(test_df) < MIN_ROWS:
      print(f"[model_decay] Window {idx}: skipped (min_rows={MIN_ROWS})", flush=True)
      skipped_windows.append(
        {
          "window_index": idx,
          "train_rows": len(train_df),
          "test_rows": len(test_df),
          "train_date_start": window["train_date_start"],
          "train_date_end": window["train_date_end"],
          "test_date_start": window["test_date_start"],
          "test_date_end": window["test_date_end"],
          "mae": np.nan,
          "status": "skipped",
        }
      )
      continue

    mae, train_rows, test_rows = _fit_train_and_score(train_df=train_df, test_df=test_df, target=target)
    print(f"[model_decay] Window {idx}: trained | MAE={mae:.6f}", flush=True)
    results.append(
      {
        "window_index": idx,
        "train_rows": train_rows,
        "test_rows": test_rows,
        "train_date_start": window["train_date_start"],
        "train_date_end": window["train_date_end"],
        "test_date_start": window["test_date_start"],
        "test_date_end": window["test_date_end"],
        "mae": mae,
        "status": "trained",
      }
    )

  all_rows = skipped_windows + results
  summary_df = pd.DataFrame(all_rows).sort_values("window_index").reset_index(drop=True)

  logs_dir = Path(__file__).resolve().parent / "logs"
  logs_dir.mkdir(parents=True, exist_ok=True)
  date_prefix = pd.Timestamp.now().strftime("%Y-%m-%d")
  log_path = logs_dir / f"{date_prefix} {target} model_decay.log"
  png_path = logs_dir / f"{date_prefix} {target} model_decay.png"

  table_df = summary_df.copy()
  for col in ["train_date_start", "train_date_end", "test_date_start", "test_date_end"]:
    table_df[col] = table_df[col].apply(lambda x: x.strftime("%Y-%m-%d") if pd.notna(x) else "-")
  table_df["mae"] = table_df["mae"].map(lambda x: f"{x:.6f}" if pd.notna(x) else "-")

  status_label = "WARNING" if (summary_df["status"] == "trained").any() and (summary_df["status"] == "skipped").any() else "INFO"
  first_window = summary_df[summary_df["window_index"] == 1]
  first_window_decay_warning = (
    (not first_window.empty)
    and (first_window.iloc[0]["status"] == "trained")
    and pd.notna(first_window.iloc[0]["mae"])
    and float(first_window.iloc[0]["mae"]) < MAE_THRESHOLD
  )
  first_line = "WARNING: model decay" if first_window_decay_warning else f"{status_label} model_decay"
  log_lines = [
    first_line,
    f"target={target}",
    f"window_months={window_months}",
    f"min_rows={MIN_ROWS}",
    f"mae_threshold={MAE_THRESHOLD}",
    f"min_analysis_date={MIN_ANALYSIS_DATE.strftime('%Y-%m-%d')}",
    f"dataset_rows={len(df)}",
    f"dataset_date_start={df['fecha'].min().strftime('%Y-%m-%d')}",
    f"dataset_date_end={df['fecha'].max().strftime('%Y-%m-%d')}",
    "",
    table_df.to_string(index=False),
  ]
  log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
  print(f"[model_decay] Log written to: {log_path}", flush=True)

  plotted = summary_df[summary_df["status"] == "trained"].copy().sort_values("window_index")
  if not plotted.empty:
    plotted["test_label"] = plotted["test_date_end"].dt.strftime("%Y-%m-%d")
    plt.figure(figsize=(10, 5))
    plt.plot(plotted["test_date_end"], plotted["mae"], marker="o")
    plt.title(f"Model decay MAE over time ({target})")
    plt.xlabel("Test window end date")
    plt.ylabel("MAE")
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()
    print(f"[model_decay] Plot written to: {png_path}", flush=True)
  else:
    raise ValueError("No se pudo entrenar ningun window con al menos 100 filas en train y test.")

  print("[model_decay] Finished successfully.", flush=True)

  return {
    "log_path": str(log_path),
    "png_path": str(png_path),
    "trained_windows": int((summary_df["status"] == "trained").sum()),
    "skipped_windows": int((summary_df["status"] == "skipped").sum()),
    "results": summary_df.to_dict(orient="records"),
  }


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--target", type=str, default=DEFAULT_TARGET)
  parser.add_argument("--window_months", type=int, default=WINDOW_MONTHS)
  args = parser.parse_args()

  output = run_model_decay(target=args.target, window_months=args.window_months)
  print(output)
