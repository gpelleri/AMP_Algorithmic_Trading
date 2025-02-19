import yfinance as yf
import pandas as pd


class DataHandler:
    def __init__(self, ticker, start_date, end_date):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date

    def fetch_data(self):
        data = yf.download(self.ticker, start=self.start_date, end=self.end_date)
        # deal with latest version of yfinance where data is now multi-index dataframe
        if isinstance(data.columns, pd.MultiIndex):
            data = data.xs(key=self.ticker, axis=1, level=1)  # Extract relevant ticker's data

        data = data['Close'].dropna()
        return data
