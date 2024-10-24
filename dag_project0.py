import json
import logging
from datetime import time, timedelta
from io import StringIO

import mlflow
import os

import pandas as pd
from airflow.models import DAG, Variable
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.utils.dates import days_ago

from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger()
LOG.addHandler(logging.StreamHandler())

S3_CONN = "s3_connection"
S3_PREFIX = "KseniaLushcheva"
S3_BUCKET = Variable.get("S3_BUCKET")

DAG_NAME = "project_dag"
DEFAULT_ARGS = {
    "owner": "Ksenia Lushcheva",
    "retries": 3,
    "retry_delay": timedelta(minutes=1),
}

MODELS = {
    "random_forest": RandomForestRegressor(),
    "hist_gradientboost": HistGradientBoostingRegressor(),
    "decision_tree": DecisionTreeRegressor(),
}

EXPERIMENT_NAME = "ksenia_lushcheva"
PARENT_RUN_NAME = "supostatka"


def configure_mlflow():
    for key in [
        "MLFLOW_TRACKING_URI",
        "AWS_ENDPOINT_URL",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_DEFAULT_REGION",
    ]:
        os.environ[key] = Variable.get(key)


def get_experiment_id(experiment_name: str) -> str:
    client = MlflowClient()
    try:
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment:
            return experiment.experiment_id
        else:
            exp = client.create_experiment(
                name=experiment_name, artifact_location=f"s3://{S3_BUCKET}")
            return exp.experiment_id
    except Exception as e:
        print(f"Error getting experiment ID: {e}")
        raise


def save_data_to_s3(data: pd.DataFrame, s3_hook: S3Hook, path: str):
    try:
        csv_buffer = StringIO()
        data.to_csv(csv_buffer, index=False)

        s3_hook.load_string(
            string_data=csv_buffer.getvalue(),
            bucket_name=S3_BUCKET,
            key=path,
            replace=True,
        )
    except Exception as e:
        print(f"Error loading data to S3: {e}")
        raise


def load_data_from_s3(s3_hook: S3Hook, path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(s3_hook.download_file(key=path, bucket_name=S3_BUCKET))
    except Exception as e:
        print(f"Error loading data from S3: {e}")
        raise


def scale_data(X_train, X_val, X_test):
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    return X_train_scaled, X_val_scaled, X_test_scaled


def init(**kwargs):
    try:
        timestamp = time()
        kwargs["ti"].xcom_push(key="init_metrics", value={"timestamp": timestamp})

        configure_mlflow()

        exp_id = get_experiment_id(EXPERIMENT_NAME)

        with mlflow.start_run(
                run_name=PARENT_RUN_NAME,
                experiment_id=exp_id,
                description="parent",
        ) as parent_run:
            kwargs["ti"].xcom_push(
                key="exp_info",
                value={
                    "exp_name": EXPERIMENT_NAME,
                    "exp_id": exp_id,
                    "parent_run_name": PARENT_RUN_NAME,
                    "parent_run_id": parent_run.info.run_id,
                },
            )
    except Exception as e:
        print(f"Error during initialization: {e}")
        raise


def get_data(**kwargs):
    time_start = time()

    data = load_diabetes()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target

    time_end = time()

    metrics = {
        "time_start": time_start,
        "time_end": time_end,
        "dataset_size": df.shape,
    }
    kwargs["ti"].xcom_push(
        key="data_metrics",
        value=metrics,
    )

    s3 = S3Hook(S3_CONN)
    save_data_to_s3(
        data=df,
        s3_hook=s3,
        path=f"{S3_PREFIX}/datasets/data.csv",
    )


def prepare_data(**kwargs):
    start_time = time()

    s3 = S3Hook(S3_CONN)
    data = load_data_from_s3(s3_hook=s3, path=f"{S3_PREFIX}/datasets/data.csv")

    X = data.drop(columns=["target"])
    y = data["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

    X_train_scaled, X_val_scaled, X_test_scaled = scale_data(X_train, X_val, X_test)

    X_train_scaled["target"] = y_train
    X_val_scaled["target"] = y_val

    end_time = time()

    metrics = {
        "start_time": start_time,
        "end_time": end_time,
        "features": X.columns.tolist(),
    }

    kwargs["ti"].xcom_push(
        key="prepare_metrics",
        value=metrics,
    )

    s3 = S3Hook(S3_CONN)

    save_data_to_s3(data=X_train_scaled, s3_hook=s3, path=f"{S3_PREFIX}/datasets/data_train.csv")
    save_data_to_s3(data=X_val_scaled, s3_hook=s3, path=f"{S3_PREFIX}/datasets/data_val.csv")
    save_data_to_s3(data=X_test_scaled, s3_hook=s3, path=f"{S3_PREFIX}/datasets/data_test.csv")


def train_and_log_model(model, model_name, **kwargs):
    time_start = time()

    exp_info = kwargs["ti"].xcom_pull(key="exp_info")

    s3 = S3Hook(S3_CONN)
    data_train = load_data_from_s3(s3_hook=s3, path=f"{S3_PREFIX}/datasets/data_train.csv")
    data_val = load_data_from_s3(s3_hook=s3, path=f"{S3_PREFIX}/datasets/data_val.csv")
    X_test = load_data_from_s3(s3_hook=s3, path=f"{S3_PREFIX}/datasets/data_test.csv")

    X_train = data_train.drop(columns=["target"])
    y_train = data_train["target"]
    model.fit(X_train, y_train)

    time_end = time()

    metrics = {
        "start_time": time_start,
        "end_time": time_end,
    }
    kwargs["ti"].xcom_push(
        key="train_metrics",
        value=metrics,
    )

    eval_df = data_val.copy()

    with mlflow.start_run(
            run_name=model_name,
            exp_id=exp_info["exp_id"],
            nested=True,
            parent_run_id=exp_info["parent_run_id"],
    ) as child_run:
        signature = infer_signature(X_test, model.predict(X_test))
        model_info = mlflow.sklearn.log_model(
            model,
            model_name,
            signature=signature,
        )

        mlflow.evaluate(
            model=model_info.model_uri,
            data=eval_df,
            targets="target",
            model_type="regressor",
            evaluators=["default"],
        )

    kwargs["ti"].xcom_push(
        key=f"model_info-{model_name}",
        value={
            "model_uri": model_info.model_uri,
            "model_name": model_name,
        },
    )


def save_results(models, **kwargs) -> None:
    init_metrics = kwargs["ti"].xcom_pull(key="init_metrics")
    data_metrics = kwargs["ti"].xcom_pull(key="data_metrics")
    prepare_metrics = kwargs["ti"].xcom_pull(key="prepare_metrics")
    train_metrics = kwargs["ti"].xcom_pull(key="train_metrics")
    mlflow_exp_info = kwargs["ti"].xcom_pull(key="exp_info")

    for model in models:
        model_info = kwargs["ti"].xcom_pull(key=f"model_info-{model}")

        all_metrics = {
            "dag_name": DAG_NAME,
            "mlflow_exp_info": mlflow_exp_info,
            "model_info": model_info,
            "init": init_metrics,
            "get_data": data_metrics,
            "prepare_data": prepare_metrics,
            "train_model": train_metrics,
        }

        s3 = S3Hook(S3_CONN)
        path = f"{S3_PREFIX}/{model_info['model_name']}/results/metrics.json"

        s3.load_string(
            string_data=json.dumps(all_metrics),
            bucket_name=S3_BUCKET,
            key=path,
            replace=True,
        )


dag = DAG(
    DAG_NAME,
    tags=["mlops"],
    default_args=DEFAULT_ARGS,
    schedule_interval="0 1 * * *",
    start_date=days_ago(1),
)

task_init = PythonOperator(
    task_id="init",
    python_callable=init,
    dag=dag,
)

task_get_data = PythonOperator(
    task_id="get_data",
    python_callable=get_data,
    provide_context=True,
    dag=dag,
)

task_prepare_data = PythonOperator(
    task_id="prepare_data",
    python_callable=prepare_data,
    provide_context=True,
    dag=dag,
)

training_model_tasks = [
    PythonOperator(
        task_id=f"train_model_{model_name}",
        python_callable=train_and_log_model,
        provide_context=True,
        op_kwargs={"model": model_instance, "model_name": model_name},
        dag=dag,
    )
    for model_name, model_instance in MODELS.items()
]

task_save_results = PythonOperator(
    task_id="save_results",
    python_callable=save_results,
    provide_context=True,
    dag=dag,
    op_kwargs={"models_names": MODELS.keys()},
)

(
        task_init
        >> task_get_data
        >> task_prepare_data
        >> training_model_tasks
        >> task_save_results
)
