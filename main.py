from strategies.technical_nn import TechnicalNNStrategy
from utils.backtest import Backtest
from utils.data_handler import DataHandler


if __name__ == "__main__":
    # List of S&P 500 stocks to analyze
    tickers = ['AAPL']
    start_date = '2020-01-01'
    end_date = '2024-01-01'

    # Store results for comparison
    results_price = {}
    results_return = {}

    for ticker in tickers:
        print(f"\nAnalyzing {ticker} - Return Prediction")

        # Load data
        data_handler = DataHandler(ticker=ticker, start_date=start_date, end_date=end_date)
        data = data_handler.fetch_data()

        # Initialize and run strategy
        strategy = TechnicalNNStrategy(prediction_type='return', n_splits=5, epochs=50)
        backtest = Backtest(data, strategy)
        results = backtest.run()

        # Store results
        results_return[ticker] = {
            'backtest_results': results,
            'cv_scores': strategy.cv_scores
        }

        # Plot results
        strategy.plot_signals(data)

