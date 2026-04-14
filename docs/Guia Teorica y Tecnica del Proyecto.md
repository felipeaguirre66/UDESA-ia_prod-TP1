# Guia Teorica y Tecnica del Proyecto

## Indice

1. [Que es MLOps y por que importa](#1-que-es-mlops-y-por-que-importa)
2. [Arquitectura general del sistema](#2-arquitectura-general-del-sistema)
3. [Apache Airflow - Orquestacion](#3-apache-airflow---orquestacion)
4. [MLflow - Experiment Tracking y Model Registry](#4-mlflow---experiment-tracking-y-model-registry)
5. [Feast - Feature Store](#5-feast---feature-store)
6. [FastAPI - API REST](#6-fastapi---api-rest)
7. [El modelo de ML](#7-el-modelo-de-ml)
8. [Docker y Docker Compose](#8-docker-y-docker-compose)
9. [Flujo completo de datos](#9-flujo-completo-de-datos)
10. [Decisiones de diseno y trade-offs](#10-decisiones-de-diseno-y-trade-offs)
11. [Ejemplos practicos de operaciones](#11-ejemplos-practicos-de-operaciones)
12. [Glosario](#12-glosario)

---

## 1. Que es MLOps y por que importa

### El problema

Entrenar un modelo de Machine Learning en un notebook es solo una pequeña parte del trabajo. En la vida real, la mayor parte del esfuerzo esta en todo lo que rodea al modelo:

- Como obtengo los datos de forma automatica y confiable?
- Como se que version del modelo esta en produccion?
- Como comparo el modelo de hoy con el de ayer?
- Como hago para que otros sistemas consuman mis predicciones?
- Si el modelo empeora, como me doy cuenta? Como vuelvo atras?

**MLOps** (Machine Learning Operations) es el conjunto de practicas que responde a estas preguntas. Es la interseccion entre Machine Learning, DevOps e Ingenieria de Datos.

### Analogia

Pensa en un restaurante. El chef (Data Scientist) crea la receta (modelo). Pero para que el restaurante funcione necesitas: proveedores de ingredientes (data pipeline), una cocina organizada (feature store), un sistema para que los mozos tomen pedidos (API), un registro de recetas (model registry), y alguien que coordine todo (orquestador). MLOps es armar todo eso.

### En este proyecto

Este trabajo NO pide que el modelo sea super preciso. Pide que todo el sistema alrededor funcione profesionalmente. El modelo (Random Forest) es simple a proposito, el foco esta en la infraestructura.

---

## 2. Arquitectura general del sistema

El sistema tiene 4 capas principales que interactuan entre si:

```
                        +-----------------+
                        |   Datos crudos  |
                        |  (datos.gob.ar) |
                        +--------+--------+
                                 |
                                 v
+------------------------------------------------------------------+
|                    APACHE AIRFLOW (Orquestacion)                  |
|                                                                  |
|  download_data -> prepare_offline_store -> apply_feast            |
|                      -> populate_online_store -> train_model      |
+------------------------------------------------------------------+
         |                    |                        |
         v                    v                        v
+----------------+   +----------------+   +---------------------+
|  FEAST         |   |  FEAST         |   |  MLFLOW             |
|  Offline Store |   |  Online Store  |   |  Model Registry     |
|  (Parquet)     |   |  (SQLite)      |   |  Experiment Tracker |
+----------------+   +----------------+   +---------------------+
         |                    |                        |
         +--------------------+------------------------+
                              |
                              v
                    +-------------------+
                    |   FASTAPI         |
                    |   API REST        |
                    |   /api/v1/...     |
                    +-------------------+
                              |
                              v
                    +-------------------+
                    |   Usuarios /      |
                    |   Sistemas        |
                    |   externos        |
                    +-------------------+
```

### Flujo resumido

1. **Airflow** orquesta todo el pipeline: descarga datos, calcula features, los guarda en Feast, entrena el modelo y lo registra en MLflow.
2. **Feast** almacena los features pre-computados. El offline store tiene el historico (para entrenar), el online store tiene lo mas reciente (para predicciones rapidas).
3. **MLflow** guarda cada experimento de entrenamiento (parametros, metricas, artefacto del modelo) y permite saber cual modelo esta en produccion (alias "champion").
4. **FastAPI** expone una API REST que, al recibir un pedido de pronostico, busca las features en Feast, carga el modelo champion de MLflow, y devuelve la prediccion.

---

## 3. Apache Airflow - Orquestacion

### Que problema resuelve

Sin orquestacion, ejecutar un pipeline de ML implica correr scripts manualmente en orden, recordar que paso va primero, manejar errores a mano, y rezar para que nadie se olvide de un paso. Airflow automatiza todo esto.

### Conceptos clave

#### DAG (Directed Acyclic Graph)

Un DAG es un grafo dirigido sin ciclos. En terminos practicos, es la definicion de tu workflow: que tareas hay, en que orden deben ejecutarse, y que dependencias existen entre ellas.

- **Dirigido**: Las flechas van en una sola direccion (tarea A va antes que B).
- **Aciclico**: No hay loops (B no puede depender de A si A depende de B).
- **Grafo**: Es una estructura de nodos (tareas) y aristas (dependencias).

#### Task (Tarea)

Cada nodo del DAG es una tarea. En nuestro caso, usamos la **TaskFlow API** de Airflow, que permite definir tareas como funciones de Python decoradas con `@task`. Por debajo, Airflow usa **XCom** para pasar datos entre tareas, pero con TaskFlow el paso de datos es transparente (como llamar funciones normales).

#### Scheduler

El scheduler de Airflow es el componente que monitorea los DAGs y decide cuando ejecutar las tareas. Puede programar ejecuciones periodicas (cada hora, cada dia, etc.) o ejecutar bajo demanda.

#### Worker

El worker es quien realmente ejecuta las tareas. En nuestro setup usamos **CeleryExecutor** con Redis como broker de mensajes, lo que permite que las tareas se ejecuten en paralelo si fuera necesario.

### Nuestra implementacion (dags/main.py)

```python
@dag(dag_id='ml_pipeline', description='Pipeline de Machine Learning con Airflow')
def ml_pipeline():
    start >> download_data_task() >> prepare_offline_store_task() >>
    apply_feast_task() >> populate_online_store_task() >> train_model_task()
```

El DAG define esta cadena:

| Tarea | Que hace | Por que |
|-------|----------|---------|
| `download_data_task` | Descarga CSV de datos.energia.gob.ar | Obtener datos actualizados del Ministerio de Energia |
| `prepare_offline_store_task` | Calcula features de ventana por pozo y guarda como parquet | Transformar datos crudos en features utiles para ML |
| `apply_feast_task` | Ejecuta `feast apply` para sincronizar el schema del feature store | Registrar la metadata de features en Feast |
| `populate_online_store_task` | Materializa la fila mas reciente de cada pozo al online store | Habilitar predicciones rapidas del mes actual |
| `train_model_task` | Entrena modelos para prod_gas y prod_pet, los registra en MLflow | Obtener un modelo actualizado con los datos mas recientes |

### Por que Airflow y no otro orquestador?

- Es el estandar de la industria para workflows de datos y ML.
- Tiene una UI web para monitorear y gestionar ejecuciones.
- Permite reintentar tareas individuales sin re-ejecutar todo el pipeline.
- Tiene un ecosistema enorme de integraciones.
- Es lo que se uso en la catedra (practicas de Clase 1).

### Alternativas y por que no las usamos

| Alternativa | Descripcion | Por que no |
|-------------|-------------|------------|
| **Prefect** | Orquestador moderno, mas "pythonico" | Menos maduro, menor ecosistema, no visto en la catedra |
| **Luigi** | Orquestador de Spotify | Mas simple, sin UI tan completa, comunidad mas chica |
| **Cron jobs** | Scheduler del sistema operativo | Sin UI, sin reintentos, sin dependencias entre tareas, sin logs centralizados |
| **Scripts manuales** | Correr cada script a mano | No escalable, propenso a errores, no reproducible |

---

## 4. MLflow - Experiment Tracking y Model Registry

### Que problema resuelve

Imaginemos que entrenamos 20 modelos con distintos hiperparametros. Como respondemos a estas preguntas?

- Cual fue el mejor modelo?
- Que hiperparametros uso?
- Puedo reproducirlo?
- Si pongo un modelo nuevo en produccion y anda peor, puedo volver al anterior?
- Que modelo esta en produccion ahora mismo?

Sin una herramienta como MLflow, la respuesta tipica es: "esta en un notebook en la compu de Juan", "creo que usamos n_estimators=100 pero no estoy seguro", "el pickle esta en una carpeta compartida". Esto es un desastre en produccion.

### Componentes de MLflow que usamos

#### 1. Experiment Tracking

Cada vez que entrenamos un modelo, MLflow registra un **Run** dentro de un **Experiment**. Cada run contiene:

- **Parametros** (`log_param`): Configuracion usada. En nuestro caso: n_estimators, random_state, target, features, data_max_date, training_date_cutoff, test_size.
- **Metricas** (`log_metric`): Resultados de la evaluacion. Logueamos: test_r2_score, test_mse, n_train_samples, n_test_samples.
- **Artefactos** (`log_model`): El modelo entrenado serializado. MLflow lo guarda y permite cargarlo despues para inferencia.

Esto garantiza **reproducibilidad**: si un modelo funciono bien, tenemos todos los datos para reproducirlo exactamente.

#### 2. Model Registry

El Model Registry es donde los modelos pasan de "artefacto de un experimento" a "modelo listo para produccion". Funciona asi:

1. Al terminar un entrenamiento, el modelo se registra con un nombre (ej: `prod_gas__random_forest`).
2. Cada registro crea una nueva **version** del modelo.
3. Podemos asignar **aliases** a versiones especificas. Usamos el alias `"champion"` para indicar cual version es la productiva.

Cuando la API necesita hacer una prediccion, carga el modelo con: `models:/prod_gas__random_forest@champion`. Esto siempre apunta al modelo productivo actual, sin hardcodear una version.

### Nuestra implementacion (src/train_model.py)

El flujo de entrenamiento es:

```
1. Leer features historicas del Feature Store (Feast offline)
2. Filtrar datos hasta la fecha de corte (training_date)
3. Split 80/20 para evaluar
4. Entrenar un modelo de evaluacion, loguear metricas
5. Re-entrenar con TODOS los datos para el modelo final
6. Guardar el modelo en MLflow
7. (Opcional) Asignar alias "champion"
```

Detalle importante del paso 5: se evalua con un split para medir performance, pero el modelo final se entrena con el 100% de los datos. Esto maximiza la capacidad predictiva del modelo productivo.

### Por que MLflow y no otra herramienta?

| Alternativa | Descripcion | Por que no |
|-------------|-------------|------------|
| **Weights & Biases (W&B)** | Tracking + visualizacion avanzada | Es SaaS (datos salen de tu infra), tiene costo, mas complejo |
| **Neptune.ai** | Tracking de experimentos | Similar a W&B, SaaS con costo |
| **DVC** | Versionado de datos y modelos con Git | Mas orientado a datos que a experimentos, curva de aprendizaje |
| **Guardar pickles en disco** | Serializar el modelo manualmente | Sin versionado, sin metricas, sin reproducibilidad |

MLflow es open-source, se puede hostear localmente (como en nuestro docker-compose), es el mas usado en la industria, y fue el que se uso en la catedra (Clase 2).

---

## 5. Feast - Feature Store

### Que problema resuelve: Training-Serving Skew

Este es uno de los problemas mas sutiles y peligrosos en ML en produccion: el **Training-Serving Skew** (desalineacion entre entrenamiento e inferencia).

**Ejemplo**: Durante entrenamiento, calculas el promedio de produccion de gas de los ultimos 10 meses como `window['prod_gas'].mean()`. Funciona perfecto. Pero cuando la API necesita predecir en tiempo real, como obtiene ese promedio? Si lo calculas distinto (quizas leyendo de otra tabla, o con un bug en el calculo), el modelo va a recibir features diferentes a las que vio durante entrenamiento, y la prediccion sera mala.

Un **Feature Store** resuelve esto centralizando la definicion y el calculo de features. El mismo feature store es la fuente de datos tanto para entrenamiento como para inferencia. Garantia de consistencia.

### Offline Store vs Online Store

| Aspecto | Offline Store | Online Store |
|---------|---------------|--------------|
| **Para que** | Entrenamiento (datos historicos masivos) | Inferencia en tiempo real (datos mas recientes) |
| **Formato** | Parquet (archivo columnar eficiente) | SQLite (base de datos para lecturas rapidas) |
| **Contenido** | Todos los features historicos por pozo y fecha | Solo la fila mas reciente por pozo |
| **Velocidad** | Mas lento (lee todo el archivo) | Muy rapido (lookup por key) |
| **Cuando se usa** | `store.get_historical_features()` | `store.get_online_features()` |

### Conceptos de Feast

#### Entity (Entidad)

La entidad es la "key" del feature store. En nuestro caso, la entidad es `idpozo` (el identificador del pozo). Cada fila de features esta asociada a un pozo y un timestamp.

```python
pozo = Entity(name="idpozo", description="Identificador unico del pozo")
```

#### Feature View (Vista de Features)

Define que features existen, sus tipos, y de donde vienen. Es como un "schema" del feature store.

```python
well_stats = FeatureView(
    name="well_stats",
    entities=[pozo],
    schema=[
        Field(name="avg_prod_gas_10m", dtype=Float32),  # Promedio produccion gas ultimos 10 meses
        Field(name="avg_prod_pet_10m", dtype=Float32),  # Promedio produccion petroleo ultimos 10 meses
        Field(name="last_prod_gas",    dtype=Float32),   # Ultima produccion de gas registrada
        Field(name="last_prod_pet",    dtype=Float32),   # Ultima produccion de petroleo registrada
        Field(name="n_readings",       dtype=Int32),     # Cantidad de lecturas en la ventana
        Field(name="tipoextraccion",   dtype=String),    # Tipo de extraccion
        # ... mas campos
    ],
    source=well_stats_source,  # Apunta al parquet
)
```

#### FileSource (Fuente de datos)

El offline store es simplemente un archivo Parquet. Feast lo lee cuando pedis features historicas.

### Nuestra implementacion

#### Preparacion del offline store (feature_store/prepare_offline_store.py)

Este script transforma datos crudos en features:

1. **Descarga** el CSV de produccion del Ministerio de Energia.
2. **Agrupa por pozo** y calcula features de ventana deslizante de 10 meses:
   - `avg_prod_gas_10m`: Promedio de produccion de gas de las ultimas 10 lecturas.
   - `avg_prod_pet_10m`: Promedio de produccion de petroleo de las ultimas 10 lecturas.
   - `last_prod_gas` / `last_prod_pet`: La produccion mas reciente.
   - `n_readings`: Cuantas lecturas hay en la ventana (idealmente 10).
3. **Crea una fila "futura"** por pozo: con fecha un mes despues de la ultima lectura, sin valores de produccion (prod_gas=None, prod_pet=None). Esta fila existe para que el online store tenga contexto para predecir el proximo mes.
4. **Guarda como Parquet**: Este archivo es el offline store de Feast.

**Limitacion**: Por restricciones de memoria en Docker, solo se guardan las ultimas 5 filas por pozo (en vez de todo el historial). Esto reduce la cantidad de datos de entrenamiento pero mantiene performance aceptable. En produccion real, no hariamos esta reduccion.

#### Poblacion del online store (feature_store/populate_online_store.py)

1. Lee el parquet (offline store).
2. Toma la ultima fila de cada pozo.
3. La escribe al online store (SQLite) usando `store.write_to_online_store()`.

Esto permite que cuando la API pide features del mes actual para un pozo, el lookup sea instantaneo.

### Por que Feast y no otra herramienta?

| Alternativa | Descripcion | Por que no |
|-------------|-------------|------------|
| **Tecton** | Feature Store enterprise | SaaS, requiere cloud (AWS/GCP), tiene costo |
| **Hopsworks** | Feature Store open-source | Mas complejo de deploy, requiere mas infra |
| **Calcular features ad-hoc** | Calcular al momento de entrenar/predecir | Riesgo de training-serving skew, no reutilizable |
| **Tabla en base de datos** | Guardar features en PostgreSQL | No tiene el concepto de offline/online, no tiene versionado temporal |

Feast es open-source, liviano, soporta offline y online stores, y fue el que se uso en la catedra (Clase 3).

---

## 6. FastAPI - API REST

### Que problema resuelve

Un modelo entrenado no sirve de nada si nadie puede usarlo. La API es el punto de contacto entre el modelo y el mundo exterior. Permite que dashboards, aplicaciones de planificacion, u otros sistemas consulten pronosticos de manera programatica.

### OpenAPI / Swagger

La especificacion OpenAPI (antes Swagger) es un estandar para describir APIs REST. FastAPI la genera automaticamente a partir del codigo Python (tipos, Pydantic models, decoradores). Esto permite:

- Documentacion interactiva auto-generada en `/docs`.
- Validacion automatica de parametros.
- Generacion de clientes en cualquier lenguaje.

### Nuestros endpoints

#### GET /api/v1/forecast

**Proposito**: Obtener el pronostico de produccion de un pozo para un rango de fechas.

**Parametros**:
- `id_well` (requerido): Identificador del pozo (ej: "96639")
- `date_start` (requerido): Fecha de inicio (YYYY-MM-DD)
- `date_end` (requerido): Fecha de fin (YYYY-MM-DD)
- `target` (opcional, default: "prod_gas"): Variable objetivo ("prod_gas" o "prod_pet")

**Respuesta**:
```json
{
  "id_well": "96639",
  "data": [
    {"date": "2025-10-01", "prod": 1523.45},
    {"date": "2025-11-01", "prod": 1487.20}
  ]
}
```

**Nota sobre `target`**: Este parametro no esta en la especificacion original del RFC, pero se agrego porque el sistema predice dos variables diferentes (gas y petroleo) y necesita saber cual devolver. Tiene un valor por defecto (`prod_gas`) para mantener compatibilidad con la spec.

#### GET /api/v1/wells

**Proposito**: Obtener el listado de pozos disponibles para una fecha dada.

**Parametros**:
- `date_query` (requerido): Fecha de consulta (YYYY-MM-DD)

**Respuesta**:
```json
[
  {"id_well": "96639"},
  {"id_well": "132879"},
  ...
]
```

### Flujo interno de una prediccion

Cuando llega un request a `/api/v1/forecast`:

1. **Validacion**: Se verifica que `id_well` sea numerico y que las fechas sean validas.
2. **Carga del modelo**: Se carga el modelo "champion" de MLflow (`models:/prod_gas__random_forest@champion`).
3. **Obtencion de features**:
   - Si es una sola fecha y coincide con el ultimo mes disponible: usa el **online store** (rapido).
   - Si no: usa el **offline store** (lee el parquet y busca las filas correspondientes).
4. **Prediccion**: Se construye el input para el modelo (con one-hot encoding de `tipoextraccion`), se predice, y se asegura que el valor sea >= 0.
5. **Respuesta**: Se formatea y devuelve como JSON.

---

## 7. El modelo de ML

### Random Forest Regressor

El modelo elegido es un **Random Forest Regressor** de scikit-learn. Es un modelo de ensemble basado en arboles de decision.

#### Como funciona (conceptualmente)

1. **Bagging**: Se crean N arboles de decision (200 en nuestro caso). Cada arbol ve un subconjunto aleatorio de los datos de entrenamiento (muestreo con reemplazo, o "bootstrap").
2. **Feature subsampling**: Ademas, en cada split de cada arbol, solo se considera un subconjunto aleatorio de features. Esto reduce la correlacion entre arboles.
3. **Promediado**: Para hacer una prediccion, cada arbol da su prediccion individual, y el resultado final es el **promedio** de todas las predicciones.

#### Por que Random Forest para este proyecto?

- **Robusto**: Funciona razonablemente bien sin mucho tuning.
- **No requiere normalizacion**: A diferencia de regresion lineal o redes neuronales, los arboles de decision no requieren que los features esten en la misma escala.
- **Interpretable**: Puedes ver la importancia de cada feature.
- **Rapido de entrenar**: Para el volumen de datos que manejamos, entrena en segundos.
- **El foco no es el modelo**: El RFC explicitamente dice que el foco es la infraestructura, no la sofisticacion del modelo. Un Random Forest es un baseline solido.

#### Alternativas y por que no

| Modelo | Por que no |
|--------|------------|
| **Regresion lineal** | Demasiado simple, no captura relaciones no lineales |
| **XGBoost / LightGBM** | Mejor performance pero mas complejidad en tuning, no aporta al foco del trabajo |
| **Redes neuronales (LSTM, etc.)** | Overkill para el problema, requiere mas datos, GPU, y tuning |
| **ARIMA / Prophet** | Modelos de series temporales, validos pero el enfoque del proyecto es diferente |

### Features del modelo

| Feature | Tipo | Descripcion | Intuicion |
|---------|------|-------------|-----------|
| `tipoextraccion` | Categorica | Tipo de extraccion del pozo (one-hot encoded) | Diferentes metodos de extraccion producen volumenes diferentes |
| `avg_prod_gas_10m` | Numerica | Promedio de produccion de gas, ultimos 10 meses | Tendencia historica reciente de produccion de gas |
| `avg_prod_pet_10m` | Numerica | Promedio de produccion de petroleo, ultimos 10 meses | Tendencia historica reciente de produccion de petroleo |
| `last_prod_gas` | Numerica | Ultima produccion de gas registrada | Valor mas cercano en el tiempo |
| `last_prod_pet` | Numerica | Ultima produccion de petroleo registrada | Valor mas cercano en el tiempo |
| `n_readings` | Numerica | Cantidad de lecturas en la ventana de 10 meses | Indicador de cuanta informacion hay disponible |

### Metricas de evaluacion

| Metrica | Que mide | Interpretacion |
|---------|----------|----------------|
| **R2 (R-squared)** | Proporcion de varianza explicada por el modelo | 1.0 = perfecto, 0.0 = predice la media, <0 = peor que la media |
| **MSE (Mean Squared Error)** | Error cuadratico medio | Menor es mejor. Penaliza errores grandes mas que los chicos |

### Estrategia de entrenamiento

1. **Split para evaluacion**: Se divide 80% train, 20% test. Se entrena, se evalua, se loguean metricas en MLflow.
2. **Reentrenamiento final**: Despues de evaluar, se reentrena el modelo con el **100% de los datos**. Este es el modelo que se guarda para produccion.

Por que reentrenar con todo? Porque el split solo sirve para estimar la performance. Para el modelo productivo, queremos usar toda la informacion disponible para maximizar la capacidad predictiva.

---

## 8. Docker y Docker Compose

### Que problema resuelve

"En mi maquina funciona" es el clasico problema de desarrollo. Docker resuelve esto empaquetando la aplicacion con todas sus dependencias en un contenedor aislado que funciona igual en cualquier maquina.

### Conceptos clave

- **Imagen**: Un "snapshot" de un sistema con todo lo necesario (OS base, Python, librerias, codigo).
- **Contenedor**: Una instancia ejecutandose de una imagen.
- **Docker Compose**: Herramienta para definir y ejecutar multiples contenedores como un sistema. Nuestro `docker-compose.yaml` define 10+ servicios que levantan juntos.

### Nuestros servicios

| Servicio | Imagen/Build | Puerto | Funcion |
|----------|-------------|--------|---------|
| `postgres` | postgres:16 | (interno) | Base de datos de Airflow (estado de DAGs, tareas, etc.) |
| `redis` | redis:7.2 | (interno) | Broker de mensajes para Celery (cola de tareas) |
| `mlflow` | ghcr.io/mlflow/mlflow:v3.10.1 | 9191 | Servidor de tracking de experimentos y model registry |
| `airflow-apiserver` | apache/airflow:3.1.7 | 8080 | UI web de Airflow + API |
| `airflow-scheduler` | apache/airflow:3.1.7 | (interno) | Programa y dispara ejecuciones de DAGs |
| `airflow-worker` | apache/airflow:3.1.7 | (interno) | Ejecuta las tareas realmente (con Feast instalado) |
| `airflow-dag-processor` | apache/airflow:3.1.7 | (interno) | Parsea los archivos de DAGs |
| `airflow-triggerer` | apache/airflow:3.1.7 | (interno) | Maneja tareas asincronicas |
| `forecast-api` | Build propio (Dockerfile.api) | 8000 | API REST de pronosticos |

### La solucion del conflicto Feast-Airflow

Un problema concreto que resolvimos: Feast depende de una version de `uvicorn` que es incompatible con la que necesita Airflow 3.1.7. La solucion fue instalar Feast **solo en el worker** (donde realmente se ejecutan las tareas de ML) y no en el scheduler ni el apiserver.

Esto se logra con dos variables de entorno separadas en `.env`:
- `_PIP_ADDITIONAL_REQUIREMENTS`: Para scheduler, apiserver (sin Feast).
- `_PIP_ADDITIONAL_REQUIREMENTS_WORKER`: Para worker (con Feast).

---

## 9. Flujo completo de datos

### Desde dato crudo hasta prediccion

```
1. DESCARGA
   datos.energia.gob.ar --> CSV crudo (data/dataset.csv)
   Campos: idpozo, anio, mes, prod_pet, prod_gas, prod_agua, tef,
           profundidad, tipoextraccion, etc.

2. FEATURE ENGINEERING (prepare_offline_store.py)
   CSV crudo --> Parquet con features calculados
   - Se crea columna 'fecha' a partir de anio+mes
   - Se seleccionan columnas relevantes
   - Para cada pozo:
     - Se calcula ventana deslizante de 10 meses
     - Se generan features: avg_prod_gas_10m, avg_prod_pet_10m,
       last_prod_gas, last_prod_pet, n_readings
     - Se crea una fila "futura" (un mes despues, sin produccion)
   - Se guardan las ultimas 5 filas por pozo en Parquet

3. FEAST APPLY
   Se ejecuta 'feast apply' para registrar el schema de features

4. MATERIALIZACION (populate_online_store.py)
   Parquet --> SQLite (online store)
   - Se toma la ultima fila de cada pozo
   - Se escribe al online store para consultas rapidas

5. ENTRENAMIENTO (train_model.py)
   Parquet (via Feast offline) --> Modelo serializado en MLflow
   - Se leen features historicas via store.get_historical_features()
   - Se aplica one-hot encoding a 'tipoextraccion'
   - Se evalua con split 80/20 (loguea metricas)
   - Se reentrena con 100% de datos
   - Se guarda modelo en MLflow, opcionalmente como "champion"

6. INFERENCIA (predict_model.py, via API)
   Request HTTP --> Features (Feast) + Modelo (MLflow) --> Prediccion
   - Si es el mes mas reciente: features del online store (rapido)
   - Si no: features del offline store
   - Se aplica same one-hot encoding
   - Se predice, se asegura pred >= 0
   - Se devuelve como JSON
```

### La fila "futura" - por que existe?

Cuando un usuario pide la prediccion del proximo mes para un pozo, necesitamos features de contexto (promedio de los ultimos 10 meses, ultima produccion, etc.). Estos features se calculan a partir de datos historicos. La fila "futura" tiene estos features pre-calculados pero con prod_gas=None y prod_pet=None (porque justamente es lo que queremos predecir).

Esta fila se materializa al online store, de modo que cuando la API pida features del proximo mes, ya estan calculados y listos. Sin esta fila, no habria contexto disponible para predecir hacia adelante.

---

## 10. Decisiones de diseno y trade-offs

### 1. Seleccion del modelo champion

**Decision**: El modelo champion se marca manualmente con `--save_as_champion true` al entrenar.

**Por que**: Es simple, explicito y seguro. Sabemos exactamente que version esta en produccion.

**Trade-off**: Requiere coordinacion manual. Una mejora futura seria comparar automaticamente las metricas del nuevo modelo con el champion actual, y solo reemplazarlo si el nuevo es mejor.

### 2. Limitacion de 5 filas por pozo en el offline store

**Decision**: Se guardan solo las ultimas 5 filas por pozo en el parquet, en lugar de todo el historial.

**Por que**: Limitaciones de memoria en Docker Desktop. Con el historial completo, el proceso agota memoria. Esta limitacion reduce el volumen de datos pero mantiene la performance aceptable.

**Trade-off**: Se pierden datos historicos para entrenar. En un entorno real con mas recursos (servidor, cluster, cloud), guardariamos todo el historial.

### 3. Online store con una sola fila por pozo

**Decision**: El online store materializa solo la fila mas reciente de cada pozo.

**Por que**: El online store es para predicciones rapidas del proximo mes (caso de uso mas comun). Para consultas historicas se usa el offline store. Esto mantiene el online store liviano.

**Trade-off**: Predicciones de meses anteriores requieren acceso al offline store (mas lento que SQLite). Pero como son menos frecuentes, es un trade-off aceptable.

### 4. Parametro `target` en la API

**Decision**: Se agrego un parametro opcional `target` (prod_gas o prod_pet) al endpoint `/api/v1/forecast`.

**Por que**: El sistema predice dos variables diferentes. Necesitamos un parametro para indicar cual se quiere devolver. Tiene valor por defecto, asi que es backward-compatible con la spec del RFC.

**Trade-off**: No esta exactamente en la spec, pero la spec dice "sujeto a cambios futuros". El parametro tiene default, asi que sin pasarlo se comporta como la spec define.

### 5. Estrategia de entrenamiento: Eval con split, luego retrain con 100%

**Decision**: Primero se evalua con split 80/20 (para medir performance). Luego se reentrena con todos los datos para el modelo final.

**Por que**: El split solo sirve para estimar metricas. Para el modelo productivo, queremos maximizar la capacidad predictiva usando toda la informacion disponible.

**Trade-off**: El modelo final no tiene metricas propias (porque se entreno con todos los datos). Pero las metricas del split quedan logueadas en MLflow como referencia.

### 6. Feast aislado solo en el worker (Resolucion de conflicto de dependencias)

**Decision**: Feast se instala unicamente en el contenedor `airflow-worker`, no en scheduler ni apiserver.

**El problema**: `Feast` depende de `uvicorn<0.37`, pero `Airflow 3.1.7` requiere `uvicorn>=0.37.0`. Ambos no pueden coexistir en el mismo contenedor.

**La solucion**: Usar dos variables de entorno diferentes en `.env`:
- `_PIP_ADDITIONAL_REQUIREMENTS`: Para scheduler, apiserver (sin Feast)
- `_PIP_ADDITIONAL_REQUIREMENTS_WORKER`: Para worker (con Feast)

El worker es donde realmente se ejecutan las tareas de ML (descarga datos, calcula features, entrena modelos), asi que Feast solo se necesita ahi. El scheduler y apiserver no usan Feast directamente.

---

## 10. Ejemplos practicos de operaciones

### Hacer una prediccion (con el script)

Utiles para testing o automatizacion. Nota: Es preferible usar la API REST (`/api/v1/forecast`) en produccion.

```bash
# Prediccion para el mes actual (usa online store - rapido)
docker exec tp1-airflow-worker-1 python /opt/airflow/src/predict_model.py \
  --target prod_gas --id_well 96639 --date_start 2026-03-31 --date_end 2026-03-31

# Prediccion para un rango historico (usa offline store - mas lento)
docker exec tp1-airflow-worker-1 python /opt/airflow/src/predict_model.py \
  --target prod_gas --id_well 96639 --date_start 2024-10-01 --date_end 2024-12-01
```

**Que sucede internamente**:
1. El script carga el modelo champion desde MLflow
2. Busca las features en Feast (online si es mes reciente, offline si es historico)
3. Aplica transformaciones (one-hot encoding de tipoextraccion)
4. Predice y devuelve el resultado

### Re-entrenar el modelo para una fecha especifica

Util para experimentar y evaluar como habria performado el modelo en el pasado (backtesting).

```bash
# Entrenar pero NO marcar como champion (para pruebas/experimentacion)
docker compose exec airflow-worker python src/train_model.py \
  --target prod_gas --training_date 2024-06-01 --save_as_champion false

# Entrenar y marcar como champion (reemplaza el modelo productivo - usarlo con cuidado)
docker compose exec airflow-worker python src/train_model.py \
  --target prod_pet --training_date 2024-06-01 --save_as_champion true
```

**Parametros**:
- `--target`: prod_gas o prod_pet
- `--training_date`: Fecha de corte. El modelo solo ve datos anteriores a esta fecha.
- `--save_as_champion`: true para reemplazar el modelo productivo, false para solo experimentar

**Flujo de entrenamiento**:
1. Lee features historicas del offline store (filtrado por training_date)
2. Split 80/20 para evaluar performance
3. Loguea metricas en MLflow
4. Reentrena con 100% de datos
5. Registra el modelo en MLflow
6. (Opcional) Asigna alias "champion" si save_as_champion=true

### Cambiar el modelo champion en MLflow

Cambiar que version del modelo esta en produccion.

**Via UI (GUI)**:
1. Ir a http://localhost:9191
2. Click en "Models" en el sidebar izquierdo
3. Seleccionar el modelo (ej: `prod_gas__random_forest`)
4. Ver lista de versiones
5. Hacer click en una version
6. Click en "Aliases" → "Add alias"
7. Escribir `champion` y confirmar

**Via API de MLflow (programatico)**:
```python
import mlflow

client = mlflow.MlflowClient("http://mlflow:9090")
client.set_registered_model_alias(
    "prod_gas__random_forest",
    "champion",
    version=5  # numero de version a marcar como champion
)
```

**Efectos**:
- La API cargara esta nueva version al recibir requests (`models:/prod_gas__random_forest@champion`)
- Las metricas de esta version quedan como referencia
- La version anterior sigue existiendo en el registry (se puede volver atras)

---

## 12. Glosario

| Termino | Definicion |
|---------|------------|
| **DAG** | Directed Acyclic Graph. Grafo de tareas con dependencias, sin ciclos |
| **Feature** | Variable de entrada del modelo. Ej: avg_prod_gas_10m |
| **Feature Store** | Sistema centralizado para almacenar y servir features |
| **Feature Engineering** | Proceso de crear features a partir de datos crudos |
| **Training-Serving Skew** | Diferencia entre los features usados en entrenamiento vs inferencia |
| **Model Registry** | Sistema para versionar y gestionar modelos de ML |
| **Champion** | El modelo que esta actualmente en produccion |
| **Experiment Tracking** | Registrar parametros, metricas y artefactos de cada entrenamiento |
| **Offline Store** | Almacen de features historicos, usado para entrenamiento |
| **Online Store** | Almacen de features recientes, usado para inferencia rapida |
| **Materializacion** | Proceso de copiar datos del offline store al online store |
| **One-hot Encoding** | Transformar una variable categorica en multiples columnas binarias |
| **Random Forest** | Modelo de ML que promedia predicciones de muchos arboles de decision |
| **R2 Score** | Metrica que mide que proporcion de la varianza es explicada por el modelo |
| **MSE** | Mean Squared Error. Error cuadratico medio entre prediccion y valor real |
| **Inference / Inferencia** | Usar un modelo entrenado para hacer predicciones con datos nuevos |
| **Pipeline** | Secuencia de pasos automatizados (ej: download -> preprocess -> train) |
| **Artifact** | Archivo producido por un entrenamiento (modelo serializado, graficos, etc.) |
| **Celery** | Sistema de colas de tareas distribuidas (usado por Airflow) |
| **Broker** | Intermediario de mensajes (Redis en nuestro caso) |
| **Parquet** | Formato de archivo columnar, eficiente para datos tabulares |
| **SQLite** | Base de datos liviana embebida en un archivo |
