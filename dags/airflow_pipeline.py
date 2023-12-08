import sys
from pathlib import Path
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

parent_dir_path = str(Path('.').parent)
sys.path.append(parent_dir_path)
from get_data import download_stock_data
from process_data import process_data
from train_test_split import prepare_and_save_data
from train_model import train_and_save_model
from inference import make_predictions

default_args = {
    'owner': 'danikin',
    'depends_on_past': False,
    'start_date': datetime(2023, 12, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'stock_prediction_pipeline',
    default_args=default_args,
    schedule_interval=timedelta(days=1)
)

download_task = PythonOperator(
    task_id='download_data',
    python_callable=download_stock_data,
    op_kwargs={'start_date': '2023-01-01', 'end_date': '{{ ds }}'},
    dag=dag,
)
process_task = PythonOperator(
    task_id='process_data',
    python_callable=process_data,
    op_kwargs={'raw_file_path': '{{ task_instance.xcom_pull(task_ids="download_data") }}'},
    dag=dag,
)
split_data_task = PythonOperator(
    task_id='split_data',
    python_callable=prepare_and_save_data,
    op_kwargs={'processed_file_path': '{{ task_instance.xcom_pull(task_ids="process_data") }}'},
    dag=dag,
)


def get_paths_and_train(**kwargs):
    file_paths = kwargs['ti'].xcom_pull(task_ids='split_data')
    X_train_path = file_paths['X_train']
    y_train_path = file_paths['y_train']
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_file_path = f'model_{timestamp}.joblib'
    train_and_save_model(X_train_path, y_train_path, model_file_path)
    return model_file_path


train_model_task = PythonOperator(
    task_id='train_model',
    python_callable=get_paths_and_train,
    provide_context=True,
    dag=dag,
)


def get_paths_and_test(**kwargs):
    file_paths = kwargs['ti'].xcom_pull(task_ids='split_data')
    model_file_path = kwargs['ti'].xcom_pull(task_ids='train_model')
    X_test_path = file_paths['X_test']
    y_test_path = file_paths['y_test']

    make_predictions(model_file_path, X_test_path, y_test_path)


test_model_task = PythonOperator(
    task_id='test_model',
    python_callable=get_paths_and_test,
    provide_context=True,
    dag=dag,
)

download_task >> process_task >> split_data_task >> train_model_task >> test_model_task
