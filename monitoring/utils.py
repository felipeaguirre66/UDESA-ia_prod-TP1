from __future__ import annotations

import pandas as pd

from src.config import DRIFT_RECENT_MONTHS, MODEL_FEATURES, PARQUET_PATH


def build_train_test_datasets(
    k_recent_months: int = DRIFT_RECENT_MONTHS,
    reference_date: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
  """Build train/test datasets for drift monitoring.

  Test data is the most recent K months from the reference date.
  Train data contains older rows.
  """
  if k_recent_months < 1:
    raise ValueError("k_recent_months debe ser >= 1")

  df = pd.read_parquet(PARQUET_PATH, columns=["fecha", *MODEL_FEATURES]).copy()
  df["fecha"] = pd.to_datetime(df["fecha"]).dt.to_period("M").dt.to_timestamp()
  df = df.dropna(subset=MODEL_FEATURES).reset_index(drop=True)

  ref_ts = pd.Timestamp(reference_date) if reference_date else pd.Timestamp.now()
  ref_month = ref_ts.to_period("M").to_timestamp()
  recent_start = (ref_month - pd.DateOffset(months=k_recent_months - 1)).to_period("M").to_timestamp()

  test_df = df[df["fecha"] >= recent_start].copy()
  train_df = df[df["fecha"] < recent_start].copy()

  if train_df.empty or test_df.empty:
    raise ValueError(
      "No se pudieron construir datasets train/test para drift. "
      f"train_rows={len(train_df)}, test_rows={len(test_df)}, recent_start={recent_start.date()}"
    )

  return train_df.reset_index(drop=True), test_df.reset_index(drop=True)
