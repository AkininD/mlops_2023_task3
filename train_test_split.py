import sys

import pandas as pd


def prepare_and_save_data(processed_file_path: str):
    data = pd.read_csv(processed_file_path)

    X = data[['Lag_1', 'Lag_2', 'Lag_3']]
    y = data['Close']

    split_index = int(len(data) * 0.8)

    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    file_names = {
        'X_train': processed_file_path.replace('.csv', '_X_train.csv'),
        'X_test': processed_file_path.replace('.csv', '_X_test.csv'),
        'y_train': processed_file_path.replace('.csv', '_y_train.csv'),
        'y_test': processed_file_path.replace('.csv', '_y_test.csv')
    }

    X_train.to_csv(file_names['X_train'], index=False)
    X_test.to_csv(file_names['X_test'], index=False)
    y_train.to_csv(file_names['y_train'], index=False)
    y_test.to_csv(file_names['y_test'], index=False)

    return file_names


if __name__ == '__main__':
    processed_file_path = sys.argv[1]
    prepare_and_save_data(processed_file_path)
