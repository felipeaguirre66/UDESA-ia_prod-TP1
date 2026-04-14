# Guion Video Demo - Entrega Parcial (16/4)

## Informacion general

- **Duracion**: 6 minutos
- **Participantes**: Todos los miembros del equipo deben participar
- **Formato**: Demo del desarrollo realizado + justificacion de decisiones de diseno
- **Criterios de evaluacion** (del RFC):
  - Cumplimiento de los requisitos
  - Capacidad para responder sobre diseno y funcionamiento
  - Supuestos y limitaciones
  - Trade-offs de las alternativas analizadas
  - Justificacion tecnica de las soluciones

---

## Estructura sugerida

### Intro (0:00 - 0:30) ~ 30 segundos

**Quien habla**: Miembro 1

> "Somos [nombres] y vamos a presentar nuestra plataforma de pronostico de produccion de hidrocarburos. El objetivo es predecir la produccion futura de gas y petroleo de pozos no convencionales usando datos publicos del Ministerio de Energia. El foco del trabajo esta en la infraestructura de MLOps, no en la sofisticacion del modelo."

**Mostrar**: Diagrama de arquitectura del RFC o uno propio con los 4 componentes (Airflow, MLflow, Feast, FastAPI).

---

### Bloque 1: Levantar el sistema con Docker Compose (0:30 - 1:15) ~ 45 segundos

**Quien habla**: Miembro 1

**Que mostrar**:
- Terminal: `docker compose up -d`
- Mostrar los contenedores corriendo: `docker ps`
- Mencionar los servicios principales

**Guion**:

> "Todo el sistema se levanta con un solo comando: `docker compose up -d`. Esto levanta 10 servicios interconectados: Airflow como orquestador con PostgreSQL y Redis, MLflow para tracking de experimentos, y nuestra API REST con FastAPI."

> "Elegimos Docker Compose porque permite reproducir el entorno exacto en cualquier maquina. Un punto a mencionar es que tuvimos que resolver un conflicto de dependencias entre Feast y Airflow: ambos dependen de versiones incompatibles de uvicorn. Lo resolvimos instalando Feast unicamente en el worker de Airflow, que es donde realmente se ejecutan las tareas de ML, no en el scheduler ni el API server."

**Requisito que cubre**: "El sistema DEBE poder levantarse localmente mediante docker-compose."

---

### Bloque 2: Pipeline de ML con Airflow (1:15 - 2:30) ~ 75 segundos

**Quien habla**: Miembro 2

**Que mostrar**:
- Abrir Airflow UI en localhost:8080
- Mostrar el DAG `ml_pipeline` y su grafo de tareas
- Ejecutar el DAG (o mostrar una ejecucion exitosa)
- Abrir los logs de alguna tarea para mostrar que funciona

**Guion**:

> "Este es nuestro DAG de Airflow. Define el pipeline completo de Machine Learning en 5 tareas secuenciales."

> "Primero, `download_data` descarga el CSV actualizado de produccion de pozos del sitio del Ministerio de Energia. Segundo, `prepare_offline_store` toma esos datos crudos y calcula features de ventana: para cada pozo, calculamos el promedio de produccion de los ultimos 10 meses, la ultima produccion registrada, y la cantidad de lecturas disponibles. Todo esto se guarda como Parquet, que es el offline store de Feast."

> "Tercero, `apply_feast` registra el schema de features. Cuarto, `populate_online_store` materializa la fila mas reciente de cada pozo al online store, una base SQLite para consultas rapidas. Y quinto, `train_model` entrena dos modelos de Random Forest: uno para produccion de gas y otro para petroleo, y los registra en MLflow."

> "Elegimos Airflow porque es el estandar de la industria para orquestacion de pipelines de datos y ML. La alternativa mas simple seria un cron job o scripts manuales, pero Airflow nos da una UI para monitorear, capacidad de reintentar tareas individuales, y logs centralizados."

**Requisito que cubre**: Pipeline funcional orquestado.

---

### Bloque 3: Feature Store con Feast (2:30 - 3:30) ~ 60 segundos

**Quien habla**: Miembro 1

**Que mostrar**:
- Abrir `feature_store/features.py` y mostrar la definicion de entidades y features
- Abrir `feature_store/prepare_offline_store.py` y mostrar el calculo de features de ventana
- Opcionalmente: mostrar el contenido del parquet o un ejemplo de `get_online_features`

**Guion**:

> "Usamos Feast como feature store. Esto resuelve uno de los problemas mas importantes en ML en produccion: el training-serving skew. Si durante entrenamiento calculamos los features de una manera y durante inferencia de otra, el modelo va a funcionar mal. Feast centraliza la definicion y el calculo de features, garantizando consistencia."

> "Tenemos un offline store en formato Parquet con el historico de features por pozo, y un online store en SQLite con solo la ultima observacion de cada pozo para predicciones rapidas del mes siguiente."

> "Los features que calculamos son: promedio de produccion en ventana de 10 meses, ultima produccion registrada, cantidad de lecturas, y tipo de extraccion. Un detalle de implementacion: creamos una fila 'futura' para cada pozo, con fecha un mes despues de la ultima lectura y sin valores de produccion. Esto permite que el online store tenga el contexto necesario para predecir hacia adelante."

> "Una limitacion actual: por restricciones de memoria en Docker, solo guardamos las ultimas 5 filas por pozo. En un entorno productivo real con mas recursos, guardariamos el historial completo."

**Requisito que cubre**: "La generacion de features DEBE quedar persistido en un feature store que sera utilizado durante la inferencia." + "El entrenamiento del modelo DEBE llevarse a cabo consumiendo del feature store."

---

### Bloque 4: Experiment Tracking con MLflow (3:30 - 4:30) ~ 60 segundos

**Quien habla**: Miembro 2

**Que mostrar**:
- Abrir MLflow UI en localhost:9191
- Mostrar los experimentos (`prod_gas__random_forest`, `prod_pet__random_forest`)
- Abrir un run y mostrar parametros logueados
- Mostrar las metricas (test_r2, test_mse)
- Ir al Model Registry y mostrar el alias "champion"

**Guion**:

> "Cada vez que entrenamos un modelo, se loguea todo en MLflow. Veamos un run: aca estan los parametros, como la cantidad de estimadores del Random Forest, el random state, la fecha de corte de datos, y las features usadas. Y aca las metricas: R cuadrado y MSE sobre el conjunto de test."

> "Esto es clave para reproducibilidad: si un modelo funciono bien, tenemos todos los datos para reproducirlo exactamente. Y si un modelo nuevo anda peor, podemos comparar las metricas."

> "En el Model Registry, cada modelo tiene versiones. Usamos el alias 'champion' para marcar cual version es la productiva. Cuando la API necesita hacer una prediccion, carga el modelo con este alias. Asi podemos cambiar el modelo en produccion sin tocar el codigo de la API."

> "Un detalle de nuestra estrategia de entrenamiento: primero evaluamos con un split 80/20 y logueamos las metricas. Luego reentrenamos con el 100% de los datos para el modelo final. Esto maximiza la capacidad predictiva del modelo productivo. Las metricas del split quedan como referencia."

**Requisito que cubre**: "Se DEBE incluir una funcionalidad que permita realizar tracking del entrenamiento." + "Se DEBEN loguear metricas y artefactos relevantes."

---

### Bloque 5: Reentrenamiento con un solo comando (4:30 - 5:00) ~ 30 segundos

**Quien habla**: Miembro 1

**Que mostrar**:
- Terminal: ejecutar el comando de reentrenamiento para una fecha especifica
```bash
docker compose exec airflow-worker python src/train_model.py \
  --target prod_gas --training_date 2024-06-01 --save_as_champion false
```
- Mostrar que aparece un nuevo run en MLflow

**Guion**:

> "El reentrenamiento se puede hacer con un solo comando para cualquier fecha. El parametro `training_date` indica la fecha de corte: el modelo solo ve datos anteriores a esa fecha, simulando un entrenamiento en el pasado. Esto es util para evaluar como habria performado el modelo en un momento dado."

> "El parametro `save_as_champion` controla si el modelo recien entrenado reemplaza al actual en produccion. Por defecto es false, para evitar reemplazar accidentalmente un buen modelo."

**Requisito que cubre**: "DEBE ser posible llevar a cabo el entrenamiento de manera repetible con un solo comando para cualquier dia dado."

---

### Bloque 6: API REST funcional (5:00 - 5:45) ~ 45 segundos

**Quien habla**: Miembro 2

**Que mostrar**:
- Abrir Swagger UI en localhost:8000/docs
- Ejecutar una consulta a `/api/v1/forecast` desde Swagger o curl
- Mostrar la respuesta JSON
- Ejecutar una consulta a `/api/v1/wells`
- Mostrar la respuesta

**Guion**:

> "Nuestra API REST esta construida con FastAPI y sigue la especificacion OpenAPI del trabajo integrador. Tiene dos endpoints."

> "`GET /api/v1/forecast` recibe un id de pozo, fecha de inicio y fin, y devuelve la produccion esperada. El campo de respuesta es `data` con objetos que tienen `date` y `prod`, exactamente como pide la spec."

> "Agregamos un parametro opcional `target` que no esta en la spec original para distinguir entre pronostico de gas y petroleo. Tiene default `prod_gas`, asi que sin pasarlo se comporta igual que la spec."

> "`GET /api/v1/wells` recibe una fecha y devuelve los pozos disponibles para ese mes."

> "Internamente, la API carga el modelo champion de MLflow y obtiene features de Feast. Para el mes mas reciente usa el online store, rapido. Para fechas historicas, el offline store."

**Requisito que cubre**: "El sistema DEBE exponer una API funcional conforme con la especificacion OpenAPI."

---

### Cierre (5:45 - 6:00) ~ 15 segundos

**Quien habla**: Miembro 1

> "En resumen, tenemos un pipeline end-to-end que se levanta con Docker Compose, orquesta con Airflow, trackea experimentos con MLflow, usa Feast como feature store para garantizar consistencia entre entrenamiento e inferencia, y expone los pronosticos via API REST. Gracias."

---

## Preguntas frecuentes que pueden surgir

Preparate para responder estas preguntas despues de la demo:

### Sobre Airflow
**P: Por que no usar un schedule automatico en el DAG?**
R: El DAG actualmente se ejecuta manualmente. El schedule automatico es requerimiento de la entrega final, no de la parcial. Igualmente, seria simple agregar un `schedule` al decorador `@dag`.

**P: Que pasa si falla una tarea?**
R: Airflow permite reintentar tareas individuales sin re-ejecutar todo el pipeline. Desde la UI se puede ver el error en los logs y relanzar solo la tarea fallida.

### Sobre MLflow
**P: Como se decide cual modelo es el champion?**
R: Actualmente es una decision manual (con `--save_as_champion true`). Una mejora seria comparar automaticamente las metricas del nuevo modelo con el champion actual y solo reemplazarlo si son mejores.

**P: Pueden volver a una version anterior del modelo?**
R: Si. El Model Registry mantiene todas las versiones. Se puede reasignar el alias "champion" a cualquier version anterior con un solo comando de la API de MLflow.

### Sobre Feast
**P: Por que offline y online store?**
R: Tienen propositos diferentes. El offline store tiene todo el historial y se usa para entrenamiento (acceso batch, velocidad no es critica). El online store tiene solo lo ultimo y se usa para inferencia en tiempo real (acceso por key, microsegundos).

**P: Que pasa si un pozo no tiene datos en el online store?**
R: La API cae al offline store. Si el pozo no existe en ningun store, devuelve un error 400 con un mensaje descriptivo.

### Sobre el modelo
**P: Por que Random Forest y no algo mas sofisticado?**
R: El RFC dice que el foco no esta en la precision del modelo sino en los procesos de ML Engineering. Random Forest es un baseline robusto que no requiere mucho tuning, entrenamiento rapido, y performance aceptable. Es la eleccion correcta para un trabajo donde la infraestructura es lo importante.

**P: Por que predecir gas y petroleo por separado?**
R: Porque son variables con comportamientos diferentes. Un unico modelo intentando predecir ambos seria mas complejo sin ventajas claras. Dos modelos especializados son mas simples y cada uno puede evaluarse independientemente.

### Sobre Docker
**P: Por que no usar Kubernetes?**
R: Kubernetes es para orquestar contenedores a escala en produccion. Para desarrollo local y este scope, Docker Compose es suficiente y mucho mas simple de configurar. El RFC pide que el sistema se levante localmente.

**P: Como resolvieron el conflicto de Feast con Airflow?**
R: Feast requiere uvicorn<0.37 pero Airflow 3.1.7 necesita uvicorn>=0.37. Lo resolvimos instalando Feast unicamente en el worker (donde corren las tareas) usando una variable de entorno separada (`_PIP_ADDITIONAL_REQUIREMENTS_WORKER`). El scheduler y apiserver no necesitan Feast.

### Sobre la API
**P: El parametro `target` no esta en la spec. Es un problema?**
R: No. Es un parametro opcional con valor por defecto `prod_gas`. Si no se pasa, la API se comporta exactamente como la spec define. Lo agregamos porque el sistema predice dos variables diferentes y necesitamos distinguirlas. La spec dice "sujeto a cambios futuros".

---

## Tips para la grabacion

1. **Tener todo listo antes de grabar**: Docker levantado, DAG ya ejecutado al menos una vez, MLflow con datos.
2. **No improvisar**: Seguir el guion pero con naturalidad. No leer textualmente.
3. **Practicar el timing**: 6 minutos pasan rapido. Hacer una pasada de practica cronometrada.
4. **Pantalla limpia**: Cerrar apps innecesarias, usar una terminal con fuente grande.
5. **Preparar los curls/commands**: Tenerlos listos en un archivo para copiar y pegar, no tipear en vivo.
6. **Dividir participacion equitativamente**: Cada miembro habla en al menos 2-3 bloques.
