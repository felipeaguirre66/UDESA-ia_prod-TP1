"""API REST de pronóstico (OpenAPI).

documentación en /docs
"""

from typing import Annotated, Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from src.config import PARQUET_PATH
from src.predict_model import predict
import ray
from ray import serve
from contextlib import asynccontextmanager


# almacena handler para comunicación con Ray Serve
forecast_handle = None

# función para iniciar y apagar Ray Serve
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Inicializa Ray en background
    ray.init(ignore_reinit_error=True)
    # Arranca Ray Serve desactivando el HTTP proxy redundante para evitar colisión de puertos
    serve.start(proxy_location="Disabled")
    
    # Desplega el modelo y guarda el handle de comunicación asincrónica
    global forecast_handle
    forecast_handle = serve.run(ForecastModel.bind())
    
    yield  # permite que FastAPI quede activo y escuchando peticiones
    
    # Apaga todo al detener la API
    serve.shutdown()
    ray.shutdown()


app = FastAPI(
    title="Oil & Gas Forecast API",
    version="1.0.0",
    description="API para consultar el listado de pozos y pronósticos de producción.",
    lifespan=lifespan,
)


class ForecastPoint(BaseModel):
    """Un punto de la serie de pronóstico (fecha y volumen esperado)."""

    date: str = Field(..., description="Fecha (YYYY-MM-DD)")
    prod: float = Field(..., description="Volumen producido esperado")


class ForecastResponse(BaseModel):
    """Cuerpo de respuesta de ``GET /api/v1/forecast`` alineado al RFC (campo ``data``)."""

    id_well: str
    data: list[ForecastPoint]


class WellItem(BaseModel):
    """Identificador de pozo en el listado de ``GET /api/v1/wells``."""

    id_well: str


TargetLiteral = Literal["prod_gas", "prod_pet"]

@serve.deployment(
    num_replicas=2,                      # 2 procesos independientes
    max_queued_requests=100              # pone en cola solicitudes en rafaga
)
class ForecastModel:
    def __init__(self):
        # aca queda para agregar recursos persistentes en producción
        pass

    async def predict(self, target: str, id_well: int, date_start: str, date_end: str) -> dict:
        """Se deriva la request de inferencia a Feast + MLflow"""
        return predict(target=target, id_well=id_well, date_start=date_start, date_end=date_end)

@app.get("/api/v1/forecast", response_model=ForecastResponse)
def get_forecast(
    id_well: Annotated[str, Query(description="Identificador del pozo")],
    date_start: Annotated[str, Query(description="Fecha de inicio (YYYY-MM-DD)")],
    date_end: Annotated[str, Query(description="Fecha de fin (YYYY-MM-DD)")],
    target: Annotated[
        TargetLiteral,
        Query(description="Variable objetivo: prod_gas o prod_pet"),
    ] = "prod_gas",
):
    """Obtiene el pronóstico de producción para un pozo y rango de fechas.

    Llama a ``predict()`` de ``predict_model`` (Feast + MLflow) y traduce la clave interna ``forecast``
    al nombre ``data`` requerido por la especificación OpenAPI del trabajo integrador.

    Raises:
        HTTPException: 400 si ``id_well`` no es numérico o si ``predict`` rechaza las fechas
            o falta contexto en el feature store.
    """
    try:
        wid = int(id_well.strip())
    except ValueError as e:
        raise HTTPException(status_code=400, detail="id_well debe ser un identificador numérico") from e

    try:
        raw = predict(target=target, id_well=wid, date_start=date_start, date_end=date_end)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    return ForecastResponse(
        id_well=str(raw["id_well"]),
        data=[ForecastPoint(date=p["date"], prod=float(p["prod"])) for p in raw["forecast"]],
    )


@app.get("/api/v1/wells", response_model=list[WellItem])
def get_wells(
    date_query: Annotated[str, Query(description="Fecha de consulta (YYYY-MM-DD)")],
):
    """Lista los pozos que tienen fila en el feature store offline para el mes de ``date_query``.

    Lee el parquet definido en ``src.config.PARQUET_PATH``, normaliza fechas al primer día del mes
    (igual que la lógica de inferencia) y devuelve ``idpozo`` únicos.

    Raises:
        HTTPException: 503 si el parquet no existe o no se puede leer; 400 si la fecha es inválida.
    """
    path = PARQUET_PATH
    if not path.is_file():
        raise HTTPException(
            status_code=503,
            detail="Feature store offline no disponible; ejecutá el pipeline (DAG ml_pipeline) primero.",
        )

    try:
        q = pd.Timestamp(date_query).to_period("M").to_timestamp()
    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=400, detail="date_query inválida; usar YYYY-MM-DD") from e

    try:
        df = pd.read_parquet(path, columns=["idpozo", "fecha"])
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"No se pudo leer el feature store: {e}") from e

    df = df.copy()
    df["fecha"] = pd.to_datetime(df["fecha"]).dt.to_period("M").dt.to_timestamp()
    mask = df["fecha"] == q
    ids = sorted(df.loc[mask, "idpozo"].dropna().astype(int).unique())
    return [WellItem(id_well=str(i)) for i in ids]
