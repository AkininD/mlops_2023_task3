import sys
from pathlib import Path

import pandas as pd


def process_data(raw_file_path):
    data = pd.read_csv(raw_file_path)
    close_prices = data[['Close']].copy()

    close_prices['Lag_1'] = close_prices['Close'].shift(1)
    close_prices['Lag_2'] = close_prices['Close'].shift(2)
    close_prices['Lag_3'] = close_prices['Close'].shift(3)
    close_prices = close_prices.dropna()

    file_name = raw_file_path.replace('.csv', '_processed.csv')
    close_prices.to_csv(file_name, index=False)
    return file_name


if __name__ == '__main__':
    raw_file_path = sys.argv[1]
    process_data(raw_file_path)
