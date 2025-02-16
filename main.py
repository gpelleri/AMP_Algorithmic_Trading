from strategies.moving_average import MovingAverage
from utils.backtest import Backtest
from utils.data_handler import DataHandler


if __name__ == "__main__":
    data_handler = DataHandler(ticker='AAPL', start_date='2022-01-01', end_date='2023-01-01')
    data = data_handler.fetch_data()

    my_strategy = MovingAverage(short_window=10, long_window=50, mode='crossover', invert_signals=False)
    backtest = Backtest(data, my_strategy)
    results = backtest.run()
    print(results)
