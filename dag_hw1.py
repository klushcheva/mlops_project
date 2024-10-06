import json
import logging
import time
from contextlib import closing
from datetime import timedelta
from io import StringIO

import pandas as pd
from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.utils.dates import days_ago
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger()
LOG.addHandler(logging.StreamHandler())

S3_CONN = "s3_connection"
S3_PREFIX = "testbucket"
BUCKET = Variable.get("S3_BUCKET")

default_args = {
    "owner": "Ksenia Lushcheva",
    "retries": 3,
    "retry_delay": timedelta(minutes=1),
}

task_defaults = {
    "provide_context": True,
}


def init_(model_name, **kwargs):
    time_start = time.time()

    metrics = {
        "time_start": time_start,
        "model_name": model_name,
    }

    kwargs["ti"].xcom_push(
        key="init_metrics",
        value=metrics,
    )


def download_data_s3(s3_hook, bucket_name, path):
    try:
        LOG.info(f"Loading data from S3: {path}")
        return pd.read_csv(s3_hook.download_file(key=path, bucket_name=bucket_name))
    except Exception as e:
        LOG.error(f"Error loading data from S3: {e}")
        raise


def upload_data_s3(s3_hook, bucket_name, file, path):
    try:
        LOG.info(f"Uploading prepared data to S3: {path}")
        s3_hook.load_string(
            string_data=file.getvalue(),
            bucket_name=bucket_name,
            key=path,
            replace=True,
        )
    except Exception as e:
        LOG.error(f"Error uploading data to S3: {e}")
        raise


def load_data(model_name, **kwargs):
    time_start = time.time()

    data = load_diabetes()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target

    time_end = time.time()

    metrics = {
        "time_start": time_start,
        "time_end": time_end,
        "dataset_size": df.shape,
    }

    kwargs["ti"].xcom_push(
        key="data_metrics",
        value=metrics,
    )

    buffer = StringIO()
    df.to_csv(buffer, index=False)

    s3 = S3Hook(S3_CONN)
    path = f"{S3_PREFIX}/{model_name}/datasets/data.csv"
    upload_data_s3(s3, BUCKET, buffer, path)


def prepare_data(model_name, **kwargs):

    def scale_data(X):
        scaler = StandardScaler()
        return pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    s3_conn = S3Hook(S3_CONN)

    time_start = time.time()

    path = f"{S3_PREFIX}/{model_name}/datasets/data.csv"
    data = download_data_s3(s3_conn, BUCKET, path)

    X = data.drop(columns=["target"])
    y = data["target"]

    X_scaled = scale_data(X)

    prepared_data = X_scaled.copy()
    prepared_data["target"] = y

    time_end = time.time()

    metrics = {
        "time_start": time_start,
        "time_end": time_end,
        "features": X.columns.tolist(),
    }

    kwargs["ti"].xcom_push(
        key="get_metrics",
        value=metrics,
    )

    with closing(StringIO()) as file:
        prepared_data.to_csv(file, index=False)
        path = f"{S3_PREFIX}/{model_name}/datasets/prepared_data.csv"
        upload_data_s3(s3_conn, BUCKET,file, path)


def train_model(model, model_name, **kwargs):
    def calculate_metrics(y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return mse, r2

    time_start = time.time()

    s3_conn = S3Hook(S3_CONN)
    path = f"{S3_PREFIX}/{model_name}/datasets/prepared_data.csv"

    df = download_data_s3(s3_conn, BUCKET, path)

    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
    LOG.info("Training the model...")
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_mse, train_r2 = calculate_metrics(y_train, train_pred)
    test_mse, test_r2 = calculate_metrics(y_test, test_pred)

    time_end = time.time()

    metrics = {
        "start_time": time_start,
        "end_time": time_end,
        "train_metrics": {"mse": train_mse, "r2": train_r2},
        "test_metrics": {"mse": test_mse, "r2": test_r2},
    }

    LOG.info(f"Training completed. Metrics: {metrics}")

    kwargs["ti"].xcom_push(
        key="model_metrics",
        value=metrics,
    )

def save_results(model_name, **kwargs):
    init_metrics = kwargs["ti"].xcom_pull(key="init_metrics")
    data_metrics = kwargs["ti"].xcom_pull(key="data_metrics")
    prepare_metrics = kwargs["ti"].xcom_pull(key="prepare_metrics")
    train_metrics = kwargs["ti"].xcom_pull(key="model_metrics")

    all_metrics = {
            "init": init_metrics,
            "get_data": data_metrics,
            "prepare_data": prepare_metrics,
            "train_model": train_metrics,
    }

    s3 = S3Hook(S3_CONN)
    path = f"{S3_PREFIX}/{model_name}/results/metrics.json"

    s3.load_string(
        string_data=json.dumps(all_metrics, indent=4),
        bucket_name=BUCKET,
        key=path,
        replace=True,
    )

def create_dag(dag_id, model_instance, model_name, schedule, default_args):
    with DAG(
            dag_id,
            tags=["airflow_project"],
            default_args=default_args,
            schedule_interval=schedule,
            start_date=days_ago(1),
    ) as dag:
        init = PythonOperator(
            task_id="init",
            python_callable=init_,
            op_kwargs={"model_name": model_name},
            doc_md="Initializing the model"
        )

        get_data_task = PythonOperator(
            task_id="load_data",
            python_callable=load_data,
            op_kwargs={"model_name": model_name},
            doc_md="Getting the data",
            provide_context = True
        )

        prepare_data_task = PythonOperator(
            task_id="prepare_data",
            python_callable=prepare_data,
            op_kwargs={"model_name": model_name},
            doc_md="Preparing the data",
            provide_context = True
        )

        train_model_task = PythonOperator(
            task_id="train_model",
            python_callable=train_model,
            op_kwargs={"model": model_instance, "model_name": model_name},
            doc_md="Training the model",
            provide_context = True
        )

        save_results_task = PythonOperator(
            task_id="save_results",
            python_callable=save_results,
            op_kwargs={"model_name": model_name},
            doc_md="Saving results",
            provide_context = True
        )

        init >> get_data_task >> prepare_data_task >> train_model_task >> save_results_task

    return dag

models = {
        "linear_regression": LinearRegression(),
        "sgd_regression": SGDRegressor(),
        "decision_tree": DecisionTreeRegressor(),
}

for model_name, model_instance in models.items():
    dag_id = f"{model_name}_dag"
    globals()[dag_id] = create_dag(
        dag_id=dag_id,
        model_name=model_name,
        model_instance=model_instance,
        schedule="0 1 * * *",
        default_args=default_args
    )