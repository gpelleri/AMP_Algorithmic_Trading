from strategies.moving_average import MovingAverage
from strategies.momentum import MomentumStrategy
from utils.backtest import Backtest
from utils.data_handler import DataHandler


if __name__ == "__main__":
    start_date = '2010-01-01'
    end_date = '2017-01-01'
    risk_free_rate = 0.02
    data_handler = DataHandler(ticker='AAPL', start_date=start_date, end_date=end_date)
    data = data_handler.fetch_data()

    # Load S&P 500 benchmark data
    benchmark_data_handler = DataHandler("^GSPC", start_date, end_date)
    benchmark_data = benchmark_data_handler.fetch_data()

    my_strategy = MomentumStrategy(period=14, overbought=70, oversold=30)
    backtest = Backtest(data, my_strategy, benchmark_data, risk_free_rate=risk_free_rate)
    results = backtest.run()
    my_strategy.plot_signals(data)
    print(results)
