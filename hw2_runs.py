from typing import Any

import mlflow
import pandas as pd

from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature

from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

EXPERIMENT_NAME = "ksenia_lushcheva"
PARENT_RUN_NAME = "supostatka"

MODELS = {
    "random_forest_regression": RandomForestRegressor(),
    "hist_gb_regression": HistGradientBoostingRegressor(),
    "decision_tree": DecisionTreeRegressor(),
}

def get_experiment_id(experiment_name: str) -> str | Any:
    client = MlflowClient()
    try:
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment:
            return experiment.experiment_id
        else:
            return client.create_experiment(
                name=experiment_name)
    except Exception as e:
        print(f"Error getting experiment ID: {e}")
        raise


def get_data():
    data = load_diabetes()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target

    return df


def scale_data(X_train, X_val, X_test):
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    return X_train_scaled, X_val_scaled, X_test_scaled


def prepare_data(data):
    X = data.drop(columns=["target"])
    y = data["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

    X_train_scaled, X_val_scaled, X_test_scaled = scale_data(X_train, X_val, X_test)

    y_train.reset_index(drop=True, inplace=True)
    y_val.reset_index(drop=True, inplace=True)

    return X_train_scaled, X_test_scaled, X_val_scaled, y_train, y_val


def train_and_log_model(model, model_name, X_train, X_test, y_train, y_val, X_val) -> None:
    model.fit(X_train, y_train)

    eval_df = X_val.copy()
    eval_df["target"] = y_val

    # Log model
    signature = infer_signature(X_test, model.predict(X_test))
    model_info = mlflow.sklearn.log_model(
        model,
        model_name,
        signature=signature,
        registered_model_name=f"sk-learn-{model_name}-model",
    )

    # Evaluate model
    mlflow.evaluate(
        model=model_info.model_uri,
        data=eval_df,
        targets="target",
        model_type="regressor",
        evaluators=["default"],
    )


if __name__ == "__main__":

    exp_id = get_experiment_id(EXPERIMENT_NAME)
    print(exp_id)
    data = get_data()

    with mlflow.start_run(
            run_name=PARENT_RUN_NAME,
            experiment_id=exp_id,
            description="parent",
    ) as parent_run:

        for model_name, model_instance in MODELS.items():

            with mlflow.start_run(
                    run_name=model_name,
                    experiment_id=exp_id,
                    nested=True,
            ) as child_run:
                try:
                    X_train, X_test, X_val, y_train, y_val = prepare_data(data=data)

                    train_and_log_model(
                        model=model_instance,
                        model_name=model_name,
                        X_train=X_train,
                        X_test=X_test,
                        y_train=y_train,
                        y_val=y_val,
                        X_val=X_val,
                    )
                except Exception as e:
                    print(f"Ошибка при обучении модели {model_name}: {e}")
                    mlflow.log_params({"error": str(e)})
                    mlflow.end_run(status="FAILED")
