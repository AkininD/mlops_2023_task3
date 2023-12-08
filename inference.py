import sys

import pandas as pd
import joblib
import mlflow
from sklearn.metrics import mean_squared_error


def make_predictions(model_file_path, X_test_path, y_test_path):
    model = joblib.load(model_file_path)
    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path)

    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test.values.ravel(), predictions)
    print(f'Test Mean Squared Error: {mse}')
    with mlflow.start_run():
        mlflow.log_metric('Test MSE', mse)

    return predictions


if __name__ == '__main__':
    model_file_path = sys.argv[1]
    X_test_path = sys.argv[2]
    y_test_path = sys.argv[3]
    make_predictions(model_file_path, X_test_path, y_test_path)
