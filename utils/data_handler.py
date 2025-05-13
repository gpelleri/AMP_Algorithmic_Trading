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

        data = data[['Close', 'High', 'Low']].dropna()
        return data

    def fetch_pb_series(self, pb_file_path):
        """
        Loads historical Price-to-Book ratio from CSV file and returns it as a time series.
        The CSV file must have columns: 'Date' and 'P/B Ratio'.
        """

        df_pb = pd.read_csv(pb_file_path)

        # Ensure 'Date' is parsed correctly and set as index
        df_pb['Date'] = pd.to_datetime(df_pb['Date'], dayfirst=True, errors='coerce')
        df_pb.set_index('Date', inplace=True)

        # Remove '_adjclose' suffix from all column names
        df_pb.columns = [col.replace('_adjclose', '') for col in df_pb.columns]

        # Extract P/B for the selected ticker
        if self.ticker not in df_pb.columns:
            raise ValueError(f"Ticker '{self.ticker}' not found in the P/B file.")

        pb_series = df_pb[[self.ticker]].dropna().rename(columns={self.ticker: 'PB_Ratio'})

        # Filter by date range
        pb_series = pb_series.loc[self.start_date:self.end_date]

        return pb_series

    def fetch_pe_series(self, pe_file_path, sheet_name="PE_ratio_hist"):
        """
        Loads historical Price-to-Earnings ratio from Excel and returns it as a time series.
        The Excel file must have a 'Dates' column and P/E ratio columns named like 'XXX UN Equity'.
        """
        xls = pd.ExcelFile(pe_file_path)
        df_pe = pd.read_excel(xls, sheet_name=sheet_name)

        # Parse and set datetime index
        df_pe['Dates'] = pd.to_datetime(df_pe['Dates'], errors='coerce')
        df_pe.set_index('Dates', inplace=True)

        # Normalize column names: strip suffixes like ' UN Equity' to get ticker
        renamed_columns = {
            col: col.split()[0] for col in df_pe.columns if isinstance(col, str)
        }
        df_pe.rename(columns=renamed_columns, inplace=True)

        if self.ticker not in df_pe.columns:
            raise ValueError(f"Ticker '{self.ticker}' not found in the P/E file. Available: {list(df_pe.columns)}")

        pe_series = df_pe[[self.ticker]].dropna().rename(columns={self.ticker: 'PE_Ratio'})
        pe_series = pe_series.loc[self.start_date:self.end_date]

        return pe_series
