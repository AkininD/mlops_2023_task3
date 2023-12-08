import sys
from pathlib import Path

import yfinance as yf


def download_stock_data(start_date, end_date, ticket='MSFT', file_stem='stock_data', dir_path='.'):
    data = yf.download(ticket, start=start_date, end=end_date)

    file_name = Path(dir_path, file_stem + '.csv')
    data.to_csv(file_name)
    return str(file_name)


if __name__ == '__main__':
    start_date = sys.argv[1]
    end_date = sys.argv[2]
    download_stock_data(start_date, end_date)
