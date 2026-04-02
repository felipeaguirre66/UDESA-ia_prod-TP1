import pickle
import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from src.config import FEATURE_STORE_REPO, PARQUET_PATH#, MODELS_DIR

def train_model():
    print("Iniciando entrenamiento del modelo...")
    from feast import FeatureStore

    store = FeatureStore(repo_path=str(FEATURE_STORE_REPO))
    
    print("Leyendo llaves de entidades desde el parquet...")
    raw_df = pd.read_parquet(PARQUET_PATH)
    entity_df = raw_df[['idpozo', 'fecha', 'prod_gas']].copy()
    entity_df['fecha'] = pd.to_datetime(entity_df['fecha'])
    entity_df = entity_df.rename(columns={'fecha': 'event_timestamp'})

    features = [
        'tipoextraccion', 'avg_prod_gas_10m', 
        'avg_prod_pet_10m',
        'last_prod_gas', 'last_prod_pet', 'n_readings'
    ]
    feast_features = [f"well_stats:{f}" for f in features]

    print("Obteniendo features históricas desde el Feature Store...")
    training_df = store.get_historical_features(
        entity_df=entity_df,
        features=feast_features,
    ).to_df()

    # Feast agrega prefijos "well_stats__", los removemos para facilitar
    training_df.columns = [c.split('__')[-1] for c in training_df.columns]
    
    # Eliminamos nulos del target (puede haber nulos por las filas "futuras" que agregamos en el offline store para online request)
    target = 'prod_gas'
    training_df = training_df.dropna(subset=[target])

    X = training_df[features]
    y = training_df[target]

    # Handle string categorical feature in training.
    X = pd.get_dummies(X, columns=['tipoextraccion'], drop_first=False)
    X = X.fillna(0)

    fecha = pd.Timestamp.now().strftime("%Y-%m-%d")
    model_type= "random_forest"
    model_name = f"{target}__{model_type}"
    model_params = {"n_estimators": 200, "random_state": 42, "n_jobs": -1}

    # 1) Validamos con train/test split y registramos metricas en MLflow.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    mlflow.set_tracking_uri("http://mlflow:9090")
    mlflow.set_experiment(model_name)

    run_name = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S") + f"__{model_name}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(model_params)
        mlflow.log_param("target", target)
        mlflow.log_param("features", ",".join(features))
        mlflow.log_param("test_size", 0.2)

        eval_model = RandomForestRegressor(**model_params)
        print("Entrenando Random Forest (split train)...")
        eval_model.fit(X_train, y_train)

        test_preds = eval_model.predict(X_test)
        test_r2 = r2_score(y_test, test_preds)
        test_mse = mean_squared_error(y_test, test_preds)

        mlflow.log_metric("test_r2_score", test_r2)
        mlflow.log_metric("test_mse", test_mse)
        mlflow.log_metric("n_train_samples", len(y_train))
        mlflow.log_metric("n_test_samples", len(y_test))

        # 2) Luego de evaluar, reentrenamos en todo el dataset para guardar el modelo final.
        model = RandomForestRegressor(**model_params)
        print("Reentrenando Random Forest con dataset completo...")
        model.fit(X, y)

        mlflow.sklearn.log_model(model, name=run_name)

    # MODELS_DIR.mkdir(parents=True, exist_ok=True)
    # model_path = MODELS_DIR / f"{fecha}__{model_name}"

    print(f"Modelo guardado en: {run_name}")
    return run_name


if __name__ == "__main__":
    train_model()