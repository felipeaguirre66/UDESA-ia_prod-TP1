from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from alibi_detect.cd import ChiSquareDrift, KSDrift

from monitoring.utils import build_train_test_datasets
from src.config import CATEGORICAL_FEATURE, DRIFT_RECENT_MONTHS, DRIFT_SIGNIFICANCE_LEVEL, NUMERICAL_FEATURES


def _ks_test(train_values: np.ndarray, test_values: np.ndarray, alpha: float) -> tuple[str, float, float]:
  detector = KSDrift(train_values.reshape(-1, 1), p_val=alpha)
  pred = detector.predict(test_values.reshape(-1, 1), drift_type="feature")
  return "alibi_ks", float(pred["data"]["p_val"][0]), float(pred["data"]["distance"][0])


def _categorical_test(train_values: pd.Series, test_values: pd.Series, alpha: float) -> tuple[str, float, float]:
  train_values = train_values.astype(str)
  test_values = test_values.astype(str)

  categories = sorted(set(train_values.unique()).union(set(test_values.unique())))
  encode = {cat: idx for idx, cat in enumerate(categories)}
  x_ref = train_values.map(encode).to_numpy(dtype=np.int64).reshape(-1, 1)
  x = test_values.map(encode).to_numpy(dtype=np.int64).reshape(-1, 1)

  detector = ChiSquareDrift(x_ref=x_ref, p_val=alpha)
  pred = detector.predict(x, drift_type="feature")
  return "alibi_chi2", float(pred["data"]["p_val"][0]), float(pred["data"]["distance"][0])


def run_data_driff(
    k_recent_months: int = DRIFT_RECENT_MONTHS,
    alpha: float = DRIFT_SIGNIFICANCE_LEVEL,
    reference_date: str | None = None,
) -> dict:
  train_df, test_df = build_train_test_datasets(
    k_recent_months=k_recent_months,
    reference_date=reference_date,
  )

  results: list[dict] = []

  for feature in NUMERICAL_FEATURES:
    method, p_val, stat = _ks_test(
      train_df[feature].to_numpy(dtype=float),
      test_df[feature].to_numpy(dtype=float),
      alpha=alpha,
    )
    results.append(
      {
        "feature": feature,
        "test": method,
        "p_value": p_val,
        "statistic": stat,
        "significant": p_val < alpha,
      }
    )

  method, p_val, stat = _categorical_test(
    train_df[CATEGORICAL_FEATURE],
    test_df[CATEGORICAL_FEATURE],
    alpha=alpha,
  )
  results.append(
    {
      "feature": CATEGORICAL_FEATURE,
      "test": method,
      "p_value": p_val,
      "statistic": stat,
      "significant": p_val < alpha,
    }
  )

  has_significant = any(row["significant"] for row in results)
  level = "WARNING" if has_significant else "INFO"

  date_prefix = pd.Timestamp.now().strftime("%Y-%m-%d")
  log_path = Path(__file__).resolve().parent / "logs" / f"{date_prefix} logile.log"
  log_path.parent.mkdir(parents=True, exist_ok=True)
  lines = [
    f"{level} data_driff",
    f"reference_date={reference_date or 'now'}",
    f"k_recent_months={k_recent_months}",
    f"alpha={alpha}",
    f"train_rows={len(train_df)}",
    f"test_rows={len(test_df)}",
    "results:",
  ]

  for row in results:
    lines.append(
      " - "
      f"feature={row['feature']}, test={row['test']}, p_value={row['p_value']:.6g}, "
      f"statistic={row['statistic']:.6g}, significant={row['significant']}"
    )

  log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

  return {
    "level": level,
    "log_path": str(log_path),
    "results": results,
  }


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--k_recent_months", type=int, default=DRIFT_RECENT_MONTHS)
  parser.add_argument("--alpha", type=float, default=DRIFT_SIGNIFICANCE_LEVEL)
  parser.add_argument("--reference_date", type=str, default=None)
  args = parser.parse_args()

  output = run_data_driff(
    k_recent_months=args.k_recent_months,
    alpha=args.alpha,
    reference_date=args.reference_date,
  )
  print(output)
