from datetime import timedelta
import sys
from pathlib import Path
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int32, String

# Ensure /opt/airflow is available when feast loads this file from /opt/airflow/feature_store.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
  sys.path.append(str(PROJECT_ROOT))

from src.config import PARQUET_PATH

# ── Entidad ──

pozo = Entity(
  name="idpozo",
  description="Identificador único del pozo de extracción",
)

# ── Fuente de datos (offline store) ──

well_stats_source = FileSource(
  path=str(PARQUET_PATH),
  timestamp_field="fecha",
)

# ── Feature View ──

well_stats = FeatureView(
  name="well_stats",
  entities=[pozo],
  schema=[
    # - Features del dataset original -
    Field(name="prod_gas",       dtype=Float32), # Target
    Field(name="prod_pet",       dtype=Float32), # Target
    Field(name="prod_agua",      dtype=Float32),
    Field(name="tef",            dtype=Float32),
    Field(name="profundidad",    dtype=Float32),
    Field(name="tipoextraccion", dtype=String),

    # - Features de ventana (últimas 10 lecturas por pozo) -
    Field(name="avg_prod_gas_10m", dtype=Float32),
    Field(name="avg_prod_pet_10m", dtype=Float32),
    Field(name="last_prod_gas",    dtype=Float32),
    Field(name="last_prod_pet",    dtype=Float32),
    Field(name="n_readings",       dtype=Int32),
  ],
  source=well_stats_source,
)