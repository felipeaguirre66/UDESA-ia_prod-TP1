
## Datos:
1. [Producción de Pozos de Gas y Petróleo No Convencional](http://datos.energia.gob.ar/dataset/c846e79c-026c-4040-897f-1ad3543b407c/archivo/b5b58cdc-9e07-41f9-b392-fb9ec68b0725)
2. [Listado de pozos cargados por empresas operadoras](http://datos.energia.gob.ar/dataset/c846e79c-026c-4040-897f-1ad3543b407c/archivo/cbfa4d79-ffb3-4096-bab5-eb0dde9a8385)

## Pasos para setear la app la primera vez:

### Entorno: OS y Docker

- **Linux / macOS**: usá `AIRFLOW_UID=$(id -u)` en el `.env` (si hay permisos raros en volúmenes, probá `50000`).

- **Windows (PowerShell/CMD)**: `AIRFLOW_UID=50000` en `.env`; ejecutá `docker compose` desde esta terminal (Docker Desktop encendido).

- **WSL2**: activá la distro en Docker Desktop → *Settings → Resources → WSL integration*; si no hay `docker`, usá PowerShell en `C:\...` del proyecto.

- **Nombre del worker**: el contenedor no siempre es `tp1-airflow-worker-1`; mirá `docker ps` y usá el nombre real.

## Pasos a seguir:

1. echo -e "AIRFLOW_UID=$(id -u)" > .env
2. echo "_PIP_ADDITIONAL_REQUIREMENTS=pandas scikit-learn mlflow pyarrow" >> .env
3. echo "_PIP_ADDITIONAL_REQUIREMENTS_WORKER=pandas scikit-learn mlflow pyarrow feast" >> .env
4. docker compose up airflow-init
5. docker compose up -d
6. ejecutar el dag `ml_pipeline` desde Airflow

Para próximos usos: `docker compose up -d`

Una vez levantado:
- Airflow: http://localhost:8080 (usuario: `airflow`, contraseña: `airflow`)
- MLflow: http://localhost:9191
- API de pronóstico (RFC): http://localhost:8000 — documentación OpenAPI (Swagger): http://localhost:8000/docs — ReDoc: http://localhost:8000/redoc

Tras el primer `docker compose up -d`, conviene construir el servicio de la API si no existe la imagen: `docker compose build forecast-api` y luego `docker compose up -d`.

## API REST (`forecast-api`)

Endpoints alineados al trabajo integrador (`GET /api/v1/forecast`, `GET /api/v1/wells`). Query opcional `target`: `prod_gas` (default) o `prod_pet`.

Ejemplos (con el stack levantado y el DAG `ml_pipeline` ejecutado al menos una vez):

```bash
curl "http://localhost:8000/api/v1/forecast?id_well=96639&date_start=2025-10-01&date_end=2025-11-01&target=prod_gas"
curl "http://localhost:8000/api/v1/wells?date_query=2025-10-01"
```

La respuesta de pronóstico usa el campo `data` (array de `date` / `prod`) según la especificación OpenAPI del enunciado.

## Pasos para predecir el rendimiento de un pozo para un rango de fechas específico:
- Ejecutar en terminal: 
    `docker exec tp1-airflow-worker-1 python /opt/airflow/src/predict_model.py --target TARGET --id_well ID_WELL --date_start DATE_START --date_end DATE_END`
    - Ejemplo de un pozo y fecha para aprovechamiento de online store: `docker exec tp1-airflow-worker-1 python /opt/airflow/src/predict_model.py --target prod_pet --id_well 96639 --date_start 2026-03-31 --date_end 2026-03-31`
    - Ejemplo de un pozo y fecha para uso de offline store: `docker exec tp1-airflow-worker-1 python /opt/airflow/src/predict_model.py --target prod_gas --id_well 96639 --date_start 2025-10-31 --date_end 2025-11-30`

## Pasos para re-entrenar un modelo de una fecha específica:
- Ejecutar en terminal: `docker compose exec airflow-worker python src/train_model.py --target TARGET --training_date FECHA --save_as_champion BOOL`
    - Ejemplo de uso: `docker compose exec airflow-worker python src/train_model.py --target prod_gas --training_date 2024-06-01 --save_as_champion false`

## Diseño
1. El modelo en producción es, por defecto, el último modelo ejecutado. Podemos verlo con el alias `champion` en el model registry de MLFLOW. Podría ser mejor agregar una regla de decisión que solo sobreescriba el champion si las métricas de evaluación son mejores que el current champion.
2. El archivo de `prepare_online_store.py` se queda sólo con las últimas 5 filas de cada pozo por que si no Docker arroja un problema de memoria completa. Aumenté la memoria alocada al servicio y aún así arroja un error. Igual, la perfomance del modelo sigue siendo decente. En un entorno real no haríamos esta reducción.
3. El online store solo contiene las features correspondientes a la última observación de cada pozo. El objetivo es que la predicción del próximo mes sea rápida de obtener ya que las features necesarias están en el online store. Esto implica que a la hora de predecir para una fecha que no sea exáctamente la siguiente a la última fecha registrada del pozo, hay que cargar las feautures del offline store. Podrían guardarse otras fechas posibles en el oline store si son frecuentemente consultadas.

### Nota de compatibilidad (Feast)
`feast` fija una version de `uvicorn` que puede romper `airflow-apiserver` (Airflow 3.1.7 requiere `uvicorn>=0.37.0`).
La solucion aplicada fue aislar `feast` solo en `airflow-worker` (donde corren las tasks) y no instalarlo globalmente en scheduler/apiserver.