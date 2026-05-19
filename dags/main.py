import pendulum

from airflow.decorators import dag, task
from airflow.operators.empty import EmptyOperator
from feature_store.prepare_offline_store import download_data, prepare_offline_store
from feature_store.populate_online_store import apply_feast, populate_online_store
from src.train_model import train_model


@dag(
    dag_id='ml_pipeline',
    description='Pipeline de Machine Learning con Airflow',
    schedule='0 0 1 * *',
    start_date=pendulum.datetime(2026, 1, 1, tz='UTC'),
    catchup=False,
)
def ml_pipeline():

    start = EmptyOperator(task_id='start')

    @task
    def download_data_task():
        download_data()

    @task
    def prepare_offline_store_task():
        prepare_offline_store()
    
    @task
    def populate_online_store_task():
        populate_online_store()

    @task
    def apply_feast_task():
        apply_feast()

    @task
    def train_model_task():
        train_model(target='prod_gas', save_as_champion=True)
        train_model(target='prod_pet', save_as_champion=True)

    @task
    def monitoring_task():
        from monitoring.data_driff import run_data_driff
        from monitoring.model_decay import run_model_decay
        run_data_driff()
        run_model_decay(target='prod_gas')
        run_model_decay(target='prod_pet')

    # Pipeline: Chain all tasks in order
    start >> download_data_task() >> prepare_offline_store_task() >> apply_feast_task() >> populate_online_store_task() >> train_model_task() >> monitoring_task()

ml_pipeline()