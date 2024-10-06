import json
import logging
import pickle
from datetime import datetime, timedelta
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from airflow import DAG
from airflow.operators.python import PythonOperator

from airflow.utils.dates import days_ago
from pydantic.v1 import BaseModel

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, median_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from airflow.models import Variable
from services.s3.connection import get_s3_hook, get_s3_resource
from sklearn.datasets import fetch_california_housing



class Connections(BaseModel):
    postgres_conn_name: str = "pg_connection"
    s3_connection: str = "s3_connection"
    bucket: str = Variable.get("S3_BUCKET")


class DagDefaultArgs(BaseModel):
    owner: str = "IgorAstashov"
    retries: int = 3
    retry_delay: timedelta = timedelta(minutes=1)
    start_date: datetime = days_ago(2)


class ModelSettings(BaseModel):
    features: List = [
        "MedInc",
        "HouseAge",
        "AveRooms",
        "AveBedrms",
        "Population",
        "AveOccup",
        "Latitude",
        "Longitude",
    ]
    target: str = "MedHouseVal"


class ModelTrainYamlSettings(BaseModel):
    default_args: DagDefaultArgs = DagDefaultArgs()
    connections: Connections = Connections()
    model: ModelSettings = ModelSettings()


settings = ModelTrainYamlSettings()

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())


def init(model_name: str) -> Dict[str, Any]:
    metrics = {"timestamp": datetime.now().strftime("%Y%m%d %H:%M"), "model": model_name}
    return metrics


def get_housing_data(**kwargs) -> dict:
    ti = kwargs["ti"]
    metrics = ti.xcom_pull(task_ids="init")

    data = fetch_california_housing()

    dataset = np.concatenate([data["data"], data["target"].reshape(-1, 1)], axis=1)
    dataset = pd.DataFrame(dataset, columns=list(data["feature_names"]) + ["MedHouseVal"])

    resource = get_s3_resource(s3_connection=settings.connections.s3_connection)
    bucket_name = settings.connections.bucket

    pickl_dump_obj = pickle.dumps(dataset)
    path = f"IgorAstashov/{metrics['model']}/datasets/california_housing.pk1"
    resource.Object(bucket_name, path).put(Body=pickl_dump_obj)

    metrics.update({
        "get_data_start": datetime.now().strftime("%Y%m%d %H:%M"),
        "get_data_end": datetime.now().strftime("%Y%m%d %H:%M"),
        "data_size": len(dataset),
    })
    return metrics


def prepare_data(**kwargs) -> Dict[str, Any]:
    ti = kwargs["ti"]
    metrics = ti.xcom_pull(task_ids="get_housing_data")

    path = f"IgorAstashov/{metrics['model']}/datasets/california_housing.pk1"
    s3 = get_s3_hook(s3_connection=settings.connections.s3_connection)
    file = s3.download_file(key=path, bucket_name=settings.connections.bucket)
    data = pd.read_pickle(file)

    x, y = data[settings.model.features], data[settings.model.target]
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.2, random_state=42)

    scaler = StandardScaler()
    x_train_fitted = scaler.fit_transform(x_train)
    x_test_fitted = scaler.transform(x_test)

    resource = get_s3_resource(s3_connection=settings.connections.s3_connection)

    for name, data in zip(
        ["x_train", "x_test", "y_train", "y_test"],
        [x_train_fitted, x_test_fitted, y_train, y_test],
    ):
        pickl_dump_obj = pickle.dumps(data)
        path = f"IgorAstashov/{metrics['model']}/datasets/{name}.pk1"
        resource.Object(settings.connections.bucket, path).put(Body=pickl_dump_obj)

    metrics.update({
        "prepare_data_start": datetime.now().strftime("%Y%m%d %H:%M"),
        "prepare_data_end": datetime.now().strftime("%Y%m%d %H:%M"),
        "features": settings.model.features,
    })

    return metrics


def train_model(model_name: str, **kwargs) -> Dict[str, Any]:
    ti = kwargs["ti"]
    metrics = ti.xcom_pull(task_ids="prepare_data")

    hook = get_s3_hook(s3_connection=settings.connections.s3_connection)

    data = {}
    for dataset in ["x_train", "x_test", "y_train", "y_test"]:
        file = hook.download_file(
            key=f"IgorAstashov/{metrics['model']}/datasets/{dataset}.pk1",
            bucket_name=settings.connections.bucket,
        )
        data[dataset] = pd.read_pickle(file)

    model_map = {
        "lr": LinearRegression(),
        "dt": DecisionTreeRegressor(),
        "rf": RandomForestRegressor(),
    }
    model = model_map[model_name]

    metrics.update({"train_start": datetime.now().strftime("%Y%m%d %H:%M")})
    model.fit(data["x_train"], data["y_train"])
    prediction = model.predict(data["x_test"])

    metrics.update({
        "train_end": datetime.now().strftime("%Y%m%d %H:%M"),
        "r2_score": r2_score(data["y_test"], prediction),
        "rmse": mean_squared_error(data["y_test"], prediction) ** 0.5,
        "mae": median_absolute_error(data["y_test"], prediction),
    })

    return metrics


def save_results(**kwargs) -> None:
    ti = kwargs["ti"]
    
    lr_metrics = ti.xcom_pull(task_ids="train_lr")
    dt_metrics = ti.xcom_pull(task_ids="train_dt")
    rf_metrics = ti.xcom_pull(task_ids="train_rf")
    
    metrics_list = [lr_metrics, dt_metrics, rf_metrics]
    
    for metrics in metrics_list:
        if metrics is not None:
            print(f"Metrics received from XCom: {metrics}")

            result_json = json.dumps(metrics, default=str)

            resource = get_s3_resource(s3_connection=settings.connections.s3_connection)

            end_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = f"IgorAstashov/{metrics['model']}/results/{end_time}_results.json"

            resource.Object(settings.connections.bucket, path).put(Body=result_json)

            logger.info(f"Results successfully saved to S3 at {path}")



for model_name, model_id in [("igor_astashov_linear_regression", "lr"), ("igor_astashov_desicion_tree", "dt"), ("igor_astashov_random_forest", "rf")]:
    with DAG(
        dag_id=model_name,
        default_args=settings.default_args.dict(),
        schedule_interval="0 1 * * *",
        catchup=False,
        tags=["mlops"],
    ) as dag:
        
        task_init = PythonOperator(
            task_id="init",
            python_callable=init,
            op_args=[model_id],
            dag=dag
        )

        task_get_data = PythonOperator(
            task_id="get_housing_data",
            python_callable=get_housing_data,
            dag=dag,
        )

        task_prepare_data = PythonOperator(
            task_id="prepare_data",
            python_callable=prepare_data,
            dag=dag,
        )

        task_train_model = PythonOperator(
            task_id=f"train_{model_id}",
            python_callable=train_model,
            op_args=[model_id],
            dag=dag,
        )

        task_save_results = PythonOperator(
            task_id="save_results",
            python_callable=save_results,
            dag=dag,
        )

        task_init >> task_get_data >> task_prepare_data >> task_train_model >> task_save_results