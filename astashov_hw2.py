import mlflow
import pandas as pd
from pandas import Series, DataFrame
from mlflow.models import infer_signature
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


NAME = "AstashovIgor"
TG_NAME = "@Astashov_I_V"

def main():
    run_experiment(
        experiment_id=get_or_create_experiment(experiment_name=NAME),
        run_name=TG_NAME,
        data=get_and_prepare_data(),
        models=dict(
            zip(["random_forest", "linear_regression", "decision_tree"],
                [RandomForestRegressor, LinearRegression, DecisionTreeRegressor]))
    )

def run_experiment(
        experiment_id: str,
        run_name: str,
        data: tuple[DataFrame, DataFrame, Series, Series],
        models: dict[str, callable]
) -> None:
    X_train, X_test, y_train, y_test = data
    with mlflow.start_run(run_name=run_name, experiment_id=experiment_id, description="parent") as parent_run:
        for model_name in models.keys():
            with mlflow.start_run(run_name=model_name, experiment_id=experiment_id, nested=True) as child_run:

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

def get_or_create_experiment(experiment_name: str) -> str:
    experiments = mlflow.search_experiments(
        filter_string=f"name = '{experiment_name}'"
    )
    return experiments[0].experiment_id if experiments else mlflow.create_experiment(NAME)


def get_and_prepare_data() -> tuple[DataFrame, DataFrame, Series, Series]:
    housing_dataset = fetch_california_housing(as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(
        housing_dataset["data"],
        housing_dataset["target"],
        test_size=0.2,
        random_state=0
    )

    scaler = StandardScaler()
    X_train_fitted = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_fitted = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    y_train.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)

    return X_train_fitted, X_test_fitted, y_train, y_test

main()
