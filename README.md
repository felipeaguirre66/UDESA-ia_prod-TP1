## Descripcion
Una plataforma end-to-end para pronosticar la produccion futura de hidrocarburos (gas y petroleo)

**Componentes principales**:
- **Apache Airflow**: Orquestacion del pipeline de ML (descarga datos, calcula features, entrena modelos)
- **Feast**: Feature Store (almacena y sirve features historicos y en tiempo real)
- **MLflow**: Tracking de experimentos y model registry (versionado de modelos)
- **FastAPI**: API REST para consultar pronosticos

## Datos:
1. [Producción de Pozos de Gas y Petróleo No Convencional](http://datos.energia.gob.ar/dataset/c846e79c-026c-4040-897f-1ad3543b407c/archivo/b5b58cdc-9e07-41f9-b392-fb9ec68b0725)
2. [Listado de pozos cargados por empresas operadoras](http://datos.energia.gob.ar/dataset/c846e79c-026c-4040-897f-1ad3543b407c/archivo/cbfa4d79-ffb3-4096-bab5-eb0dde9a8385)

## Pasos para setear la app la primera vez:

### Entorno: OS y Docker

- **Linux / macOS**: usá `AIRFLOW_UID=$(id -u)` en el `.env` (si hay permisos raros en volúmenes, probá `50000`).

- **Windows (PowerShell/CMD)**: `AIRFLOW_UID=50000` en `.env`; ejecutá `docker compose` desde esta terminal (Docker Desktop encendido).

- **WSL2**: activá la distro en Docker Desktop → *Settings → Resources → WSL integration*; si no hay `docker`, usá PowerShell en `C:\...` del proyecto.


## Pasos a seguir (Primera vez):

### 1. Preparar archivos de configuracion

```bash
# En Linux/macOS
echo -e "AIRFLOW_UID=$(id -u)" > .env
echo "_PIP_ADDITIONAL_REQUIREMENTS=pandas scikit-learn mlflow pyarrow" >> .env
echo "_PIP_ADDITIONAL_REQUIREMENTS_WORKER=pandas scikit-learn mlflow pyarrow feast" >> .env

# En Windows (PowerShell)
"AIRFLOW_UID=50000" | Out-File -Encoding UTF8 .env
"_PIP_ADDITIONAL_REQUIREMENTS=pandas scikit-learn mlflow pyarrow" | Add-Content .env
"_PIP_ADDITIONAL_REQUIREMENTS_WORKER=pandas scikit-learn mlflow pyarrow feast" | Add-Content .env
```

### 2. Inicializar y levantar los servicios

```bash
docker compose up airflow-init
docker compose up -d
docker compose build forecast-api  # opcional, puede ser necesario en algunos OS
docker compose up -d                # asegurar que forecast-api corra
```

### 3. Ejecutar el DAG

Ir a http://localhost:8080 (usuario: `airflow`, contraseña: `airflow`) y ejecutar manualmente el DAG `ml_pipeline`.

Para proximos usos solo es necesario usar `docker compose up -d`

## Acceso a los componentes

Una vez levantado, se puede acceder a os distintos componentes con:

| Componente | URL | Usuario / Contraseña | Proposito |
|-----------|-----|---------------------|----------|
| **Airflow UI** | http://localhost:8080 | airflow / airflow | Ver DAGs, ejecutar tareas, ver logs |
| **MLflow** | http://localhost:9191 | (sin autenticacion) | Ver experimentos, metricas, modelos, cambiar champion |
| **API (Swagger)** | http://localhost:8000/docs | (sin autenticacion) | Documentacion interactiva, probar endpoints |
| **API (ReDoc)** | http://localhost:8000/redoc | (sin autenticacion) | Documentacion alternativa |

## Endpoints de la API REST

### GET /api/v1/forecast

Obtiene el pronostico de produccion para un pozo y rango de fechas.

**Parametros**:
- `id_well` (requerido): Identificador del pozo (ej: "96639")
- `date_start` (requerido): Fecha de inicio (YYYY-MM-DD)
- `date_end` (requerido): Fecha de fin (YYYY-MM-DD)
- `target` (opcional, default: "prod_gas"): Variable objetivo ("prod_gas" o "prod_pet")

**Ejemplo**:
```bash
curl "http://localhost:8000/api/v1/forecast?id_well=166216&date_start=2025-12-01&date_end=2026-03-01&target=prod_gas"
```

**Respuesta**:
```json
{
  "id_well": "166216",
  "data": [
    {
      "date": "2025-12-01",
      "prod": 115.33212924999981
    },
    {
      "date": "2026-01-01",
      "prod": 125.75198084999992
    },
    {
      "date": "2026-02-01",
      "prod": 84.77035059999984
    },
    {
      "date": "2026-03-01",
      "prod": 99.98216595000031
    }
  ]
}
```

### GET /api/v1/wells

Obtiene el listado de pozos disponibles para una fecha.

**Parametros**:
- `date_query` (requerido): Fecha de consulta (YYYY-MM-DD)

**Ejemplo**:
```bash
curl "http://localhost:8000/api/v1/wells?date_query=2025-10-01"
```

**Respuesta**:
```json
[
  {"id_well": "159533"},
  {"id_well": "159558"},
  {"id_well": "163437"}
]
```

**Nota**: La respuesta de pronóstico usa el campo `data` (array de `date` / `prod`) según la especificación OpenAPI del RFC.

## Operaciones comunes

### Hacer una prediccion (via Python script)

```bash
# Nota: Reemplazar 'tp1-airflow-worker-1' con el nombre real del contenedor worker si es diferente

# Prediccion para el mes actual (usa online store - rapido)
docker exec tp1-airflow-worker-1 python /opt/airflow/src/predict_model.py \
  --target prod_gas --id_well 96639 --date_start 2026-04-01 --date_end 2026-04-30

# Prediccion para un rango historico (usa offline store)
docker exec tp1-airflow-worker-1 python /opt/airflow/src/predict_model.py \
  --target prod_gas --id_well 166216 --date_start 2025-12-01 --date_end 2026-03-01
```

Nota: Es preferible usar la API REST (`/api/v1/forecast`) en vez de ejecutar el script directamente.

### Re-entrenar el modelo para una fecha especifica

```bash
# Entrenar pero NO marcar como champion (para pruebas)
docker compose exec airflow-worker python src/train_model.py \
  --target prod_gas --training_date 2024-06-01 --save_as_champion false

# Entrenar y marcar como champion (reemplaza el modelo productivo)
docker compose exec airflow-worker python src/train_model.py \
  --target prod_pet --training_date 2024-06-01 --save_as_champion true
```

El parametro `training_date` es la fecha de corte: el modelo solo ve datos anteriores a esa fecha. Util para evaluar performance historica.

### Cambiar el modelo champion en MLflow

1. Ir a http://localhost:9191
2. Ir a "Models" → seleccionar el modelo (ej: `prod_gas__random_forest`)
3. Elegir una version de la lista
4. Click en "Aliases" → agregar/cambiar alias `champion`

Esto tambien se puede hacer programaticamente con la API de MLflow.

## Decisiones de diseno y trade-offs

Ver la seccion completa en el documento `docs/Guia Teorica y Tecnica del Proyecto.md` (seccion 10), donde se detallan:

1. Seleccion del modelo champion
2. Limitacion de 5 filas por pozo en el offline store
3. Online store con una sola fila por pozo
4. Parametro `target` en la API
5. Estrategia de entrenamiento (split + retrain)
6. Resolucion del conflicto Feast + Airflow

Ese documento contiene las justificaciones completas y trade-offs analizados para cada decision de diseno.