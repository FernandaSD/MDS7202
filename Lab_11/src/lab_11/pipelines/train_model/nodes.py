"""
This is a boilerplate pipeline 'train_model'
generated using Kedro 0.18.10
"""

import datetime
import logging
from typing import Dict

import lightgbm
import mlflow
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVR


def split_data(data: pd.DataFrame, params: Dict):

    shuffled_data = data.sample(frac=1, random_state=params["random_state"])
    rows = shuffled_data.shape[0]

    train_ratio = params["train_ratio"]
    valid_ratio = params["valid_ratio"]

    train_idx = int(rows * train_ratio)
    valid_idx = train_idx + int(rows * valid_ratio)

    assert rows > valid_idx, "test split should not be empty"

    target = params["target"]
    X = shuffled_data.drop(columns=target)
    y = shuffled_data[[target]]

    X_train, y_train = X[:train_idx], y[:train_idx]
    X_valid, y_valid = X[train_idx:valid_idx], y[train_idx:valid_idx]
    X_test, y_test = X[valid_idx:], y[valid_idx:]

    return X_train, X_valid, X_test, y_train, y_valid, y_test


def get_best_model(experiment_id):
    runs = mlflow.search_runs(experiment_id)
    print(runs)
    best_model_id = runs.sort_values("metrics.valid_mae")["run_id"].iloc[0]
    best_model = mlflow.sklearn.load_model("runs:/" + best_model_id + "/model")

    return best_model


def train_model(X_train, X_valid, y_train, y_valid):
    lr = LinearRegression()
    rfr = RandomForestRegressor(random_state=20)
    svr = SVR()
    xgbr = xgb.XGBRegressor(random_state=20)
    lgbmr = lightgbm.LGBMRegressor(random_state=20)

    fecha_hora_actual = datetime.datetime.now()
    nombre = fecha_hora_actual.strftime("%Y-%m-%d_%H-%M-%S")
    nombre_experimento = "Regresiones" + "_" + nombre
    experiment_id = mlflow.create_experiment(nombre_experimento)
    mlflow.set_experiment(nombre_experimento)

    mlflow.autolog()
    with mlflow.start_run(run_name="Linear_Regression"):
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_valid)
        mae = mean_absolute_error(y_valid, y_pred)
        mlflow.log_metric("valid_mae", mae)

    mlflow.autolog()
    with mlflow.start_run(run_name="Random_Forest_Regressor"):
        rfr.fit(X_train, y_train)
        y_pred = rfr.predict(X_valid)
        mae = mean_absolute_error(y_valid, y_pred)
        mlflow.log_metric("valid_mae", mae)

    mlflow.autolog()
    with mlflow.start_run(run_name="SVR"):
        svr.fit(X_train, y_train)
        y_pred = svr.predict(X_valid)
        mae = mean_absolute_error(y_valid, y_pred)
        mlflow.log_metric("valid_mae", mae)

    mlflow.autolog()
    with mlflow.start_run(run_name="XGBoost_Regressor"):
        xgbr.fit(X_train, y_train)
        y_pred = xgbr.predict(X_valid)
        mae = mean_absolute_error(y_valid, y_pred)
        mlflow.log_metric("valid_mae", mae)

    mlflow.autolog()
    with mlflow.start_run(run_name="LGBM_Regressor"):
        lgbmr.fit(X_train, y_train)
        y_pred = lgbmr.predict(X_valid)
        mae = mean_absolute_error(y_valid, y_pred)
        mlflow.log_metric("valid_mae", mae)

    Model = get_best_model(experiment_id)

    return Model


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    logger = logging.getLogger(__name__)
    logger.info(f"Model has a Mean Absolute Error of {mae} on test data.")
