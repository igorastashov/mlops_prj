import io
import json
import os
import pickle
from datetime import timedelta, datetime
from typing import Any, Dict

import pandas as pd
from airflow.models import DAG, Variable, TaskInstance
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.utils.dates import days_ago
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

import mlflow
from mlflow.models import infer_signature


AUTHOR_NAME = "astashov"
AUTHOR_TG_NICKNAME = "Astashov_I_V"
EMAIL = "astashovivl@gmail.com"
BUCKET = Variable.get("S3_BUCKET")
S3_HOOK = S3Hook("s3_connection")
DATASET_FILENAME = "california_housing_dataset"
PREPARED_DATASET_FILENAME = "train_test_scaled_dataset"

DEFAULT_ARGS = {
    'owner': AUTHOR_NAME,
    'email': EMAIL,
    'email_on_failure': True,
    'email_on_retry': False,
    'retry': 3,
    'retry-delay': timedelta(minutes=1)
}

models = dict(
    zip(["random_forest", "linear_regression", "decision_tree"],
        [RandomForestRegressor, LinearRegression, DecisionTreeRegressor,]))


def configure_mlflow():
    for key in [
        "MLFLOW_TRACKING_URI",
        "AWS_ENDPOINT_URL",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_DEFAULT_REGION",
    ]:
        os.environ[key] = Variable.get(key)

def get_or_create_mlflow_experiment(experiment_name: str) -> str:
    experiments = mlflow.search_experiments(
        filter_string=f"name = '{experiment_name}'"
    )
    return experiments[0].experiment_id if experiments else mlflow.create_experiment(experiment_name)

def get_current_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def create_dag(dag_id: str):

    def save_dataset_to_s3(dataset, dataset_name) -> int:
        buffer = io.BytesIO()
        pickle.dump(dataset, buffer)
        buffer.seek(0)
        n_bytes = buffer.getbuffer().nbytes
        S3_HOOK.load_file_obj(
            file_obj=buffer,
            key=f"{AUTHOR_NAME}/datasets/{dataset_name}.pkl",
            bucket_name=BUCKET,
            replace=True,
        )
        return n_bytes

    def load_dataset_from_s3(dataset_name) -> Any:
        s3_object = S3_HOOK.get_key(
            key=f"{AUTHOR_NAME}/datasets/{dataset_name}.pkl",
            bucket_name=BUCKET
        )
        buffer = io.BytesIO(s3_object.get()['Body'].read())
        dataset = pickle.load(buffer)
        return dataset

    def save_dict_as_json_to_s3(data_dict, file_name):
        buffer = io.BytesIO()
        buffer.write(json.dumps(data_dict).encode())
        buffer.seek(0)
        S3_HOOK.load_file_obj(
            file_obj=buffer,
            key=f"{AUTHOR_NAME}/results/{file_name}.json",
            bucket_name=BUCKET,
            replace=True
        )

    def init() -> Dict[str, Any]:
        configure_mlflow()
        experiment_id = get_or_create_mlflow_experiment(AUTHOR_NAME)
        with mlflow.start_run(
                run_name=AUTHOR_TG_NICKNAME,
                experiment_id=experiment_id,
                description="parent") as parent_run:
            metrics = {
                "model_names": [k for k in models.keys()],
                "pipeline_started_at": get_current_timestamp(),
                "experiment_id": experiment_id,
                "run_id": parent_run.info.run_id,
            }
            return metrics


    def get_data(ti: TaskInstance) -> Dict[str, Any]:
        metrics = ti.xcom_pull(task_ids='init')
        metrics["get_dataset_started_at"] = get_current_timestamp()

        housing_dataset = fetch_california_housing(as_frame=True)
        bytes_saved = save_dataset_to_s3(housing_dataset, DATASET_FILENAME)

        metrics["get_dataset_finished_at"] = get_current_timestamp()
        metrics["dataset_size"] = {
            "number_of_rows": housing_dataset['data'].shape[0],
            "number_of_columns": housing_dataset['data'].shape[1] + 1,
            "number_of_bytes": bytes_saved
        }
        return metrics

    def prepare_data(ti: TaskInstance) -> Dict[str, Any]:
        metrics = ti.xcom_pull(task_ids='get_data')
        metrics["prepare_data_started_at"] = get_current_timestamp()

        housing_dataset = load_dataset_from_s3(DATASET_FILENAME)
        X_train, X_test, y_train, y_test = train_test_split(
            housing_dataset["data"],
            housing_dataset["target"],
            test_size=0.3,
            random_state=42
        )
        scaler = StandardScaler()
        X_train_fitted = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
        X_test_fitted = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
        y_train.reset_index(drop=True, inplace=True)
        y_test.reset_index(drop=True, inplace=True)

        save_dataset_to_s3((X_train_fitted, X_test_fitted, y_train, y_test), PREPARED_DATASET_FILENAME)

        metrics["prepare_data_finished_at"] = get_current_timestamp()
        metrics["feature_names"] = housing_dataset['data'].columns.tolist()
        return metrics

    def train_model(model_name: str, ti: TaskInstance) -> Dict[str, Any]:
        metrics = ti.xcom_pull(task_ids='prepare_data')
        configure_mlflow()

        start_time = get_current_timestamp()
        run_id = metrics["run_id"]
        experiment_id = metrics["experiment_id"]

        X_train, X_test, y_train, y_test = load_dataset_from_s3(PREPARED_DATASET_FILENAME)
        with mlflow.start_run(run_id=run_id):
            with mlflow.start_run(
                    run_name=model_name, experiment_id=experiment_id, nested=True
            ):
                model = models[model_name]()
                model.fit(X_train, y_train)
                prediction = model.predict(X_test)
                signature = infer_signature(X_test, prediction)
                model_info = mlflow.sklearn.log_model(model, "sklearn_models", signature=signature)
                mlflow.evaluate(
                    model=model_info.model_uri,
                    data=pd.concat([X_test, y_test], axis=1),
                    targets=y_test.name,
                    model_type="regressor",
                    evaluators=["default"],
                )

        stop_time = get_current_timestamp()
        metrics['train_model_step_metrics'] = {
            "model_name": model_name,
            "started_at": start_time,
            "finished_at": stop_time,
        }
        return metrics

    def save_results(ti: TaskInstance) -> None:
        metrics_list = ti.xcom_pull(
            [f'train_model_{m_name}'
             for m_name in models.keys()]
        )
        joined_train_model_step_metrics = []
        for m in metrics_list:
            joined_train_model_step_metrics.append(m['train_model_step_metrics'])

        resulting_metrics = metrics_list[0].copy()
        resulting_metrics['train_model_step_metrics'] = joined_train_model_step_metrics
        save_dict_as_json_to_s3(resulting_metrics, 'metrics')


    dag = DAG(
        dag_id=dag_id,
        schedule_interval='0 1 * * *',
        start_date=days_ago(2),
        catchup=False,
        tags=['mlops'],
        default_args=DEFAULT_ARGS
    )

    with dag:
        task_init = PythonOperator(task_id='init', python_callable=init, dag=dag)
        task_get_data = PythonOperator(task_id='get_data', python_callable=get_data, dag=dag, provide_context=True)
        task_prepare_data = PythonOperator(task_id='prepare_data', python_callable=prepare_data, dag=dag, provide_context=True)
        tasks_train_model = [
            PythonOperator(
                task_id=f'train_model_{m_name}',
                python_callable=train_model,
                dag=dag,
                provide_context=True,
                op_kwargs={"model_name": m_name},
            )
            for m_name in models.keys()
        ]
        task_save_results = PythonOperator(task_id='save_results', python_callable=save_results, dag=dag, provide_context=True)
        task_init >> task_get_data >> task_prepare_data >> tasks_train_model >> task_save_results


create_dag(f"{AUTHOR_NAME}")