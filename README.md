
## Datos:
1. [Producción de Pozos de Gas y Petróleo No Convencional](http://datos.energia.gob.ar/dataset/c846e79c-026c-4040-897f-1ad3543b407c/archivo/b5b58cdc-9e07-41f9-b392-fb9ec68b0725)
2. [Listado de pozos cargados por empresas operadoras](http://datos.energia.gob.ar/dataset/c846e79c-026c-4040-897f-1ad3543b407c/archivo/cbfa4d79-ffb3-4096-bab5-eb0dde9a8385)

## Pasos para setear la app la primera vez:
1. echo -e "AIRFLOW_UID=$(id -u)" > .env
2. echo "_PIP_ADDITIONAL_REQUIREMENTS=pandas scikit-learn mlflow pyarrow" >> .env
3. echo "_PIP_ADDITIONAL_REQUIREMENTS_WORKER=pandas scikit-learn mlflow pyarrow feast" >> .env
4. docker compose up airflow-init
5. docker compose up -d

## Nota de compatibilidad (Feast)
`feast` fija una version de `uvicorn` que puede romper `airflow-apiserver` (Airflow 3.1.7 requiere `uvicorn>=0.37.0`).
La solucion aplicada fue aislar `feast` solo en `airflow-worker` (donde corren las tasks) y no instalarlo globalmente en scheduler/apiserver.