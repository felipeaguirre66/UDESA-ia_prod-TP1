"""API REST de pronóstico (OpenAPI).

documentación en /docs
"""

from typing import Annotated, Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from src.config import PARQUET_PATH
from src.predict_model import predict

app = FastAPI(
    title="Oil & Gas Forecast API",
    version="1.0.0",
    description="API para consultar el listado de pozos y pronósticos de producción.",
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
