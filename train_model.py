import sys

import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import mlflow
from mlflow.sklearn import log_model


def train_and_save_model(X_train_path, y_train_path, model_file_path='model.joblib'):
    with mlflow.start_run():
        X_train = pd.read_csv(X_train_path)
        y_train = pd.read_csv(y_train_path)

        n_estimators = 100
        max_depth = 5
        mlflow.log_param('n_estimators', n_estimators)
        mlflow.log_param('max_depth', max_depth)

        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train.values.ravel())
        log_model(model, 'model')
        joblib.dump(model, model_file_path)

        predictions = model.predict(X_train)
        mse = mean_squared_error(y_train.values.ravel(), predictions)
        mlflow.log_metric('Train MSE', mse)
        print(f'Train Mean Squared Error: {mse}')

    return model_file_path


if __name__ == '__main__':
    X_train_path = sys.argv[1]
    y_train_path = sys.argv[2]
    model_file_path = sys.argv[3]
    train_and_save_model(X_train_path, y_train_path, model_file_path)
