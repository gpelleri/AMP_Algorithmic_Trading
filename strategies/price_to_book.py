import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from strategies.strategy import Strategy


class PriceToBookStrategy(Strategy):
    """
    A trading strategy based on Price-to-Book (P/B) ratio.
    Generates signals based on statistical bands around the rolling mean of P/B ratio.
    """

    FREQ_MAPPING = {
        'weekly': 'W-FRI',  # Weekly data, Friday as end of week
        'monthly': 'M',  # Monthly data, last day of month
        '6month': '6M',  # 6-month data, last day of 6-month period
        'yearly': 'Y',  # Yearly data, last day of year
    }

    def __init__(self, pb_series, window: int = 5, k: float = 0.5, frequency: str = 'weekly'):
        """
        Initialize the strategy with parameters.

        Args:
            window: Rolling window size for calculating statistics
            k: Number of standard deviations for bands
            frequency: The resampling frequency for the data. Options:
                      'weekly', 'monthly', '6month', 'yearly'
        """
        if frequency not in self.FREQ_MAPPING:
            raise ValueError(f"frequency must be one of {list(self.FREQ_MAPPING.keys())}")

        self.pb_series = pb_series
        self.frequency = frequency
        self.window = window
        self.k = k
        self.signals = None
        self.rolling_mean = None
        self.upper_band = None
        self.lower_band = None

    def _resample_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Resample the input data according to the specified frequency.

        Args:
            data: Input DataFrame with datetime index

        Returns:
            Resampled DataFrame
        """
        freq = self.FREQ_MAPPING[self.frequency]
        return data.resample(freq).last().dropna()

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on P/B ratio.

        Args:
            data: DataFrame containing 'Close' and 'PB_Ratio' columns.
                 Will be resampled according to the strategy's frequency.

        Returns:
            Series of trading signals (-1, 0, 1)
        """
        # Resample the input data
        resampled_pb = self._resample_data(self.pb_series)
        pb_series = resampled_pb['PB_Ratio']

        # Calculate rolling statistics
        rolling_mean = pb_series.rolling(window=self.window).mean()
        rolling_std = pb_series.rolling(window=self.window).std()

        # Fill missing values using forward-fill and backward-fill (no look-ahead)
        rolling_mean_filled = rolling_mean.ffill().bfill()
        rolling_std_filled = rolling_std.ffill().bfill()

        # Calculate upper and lower bands
        upper_band = rolling_mean_filled + self.k * rolling_std_filled
        lower_band = rolling_mean_filled - self.k * rolling_std_filled

        # Generate low-frequency signals (e.g., weekly/monthly)
        low_freq_signals = pd.Series(0, index=pb_series.index)
        low_freq_signals[pb_series < lower_band] = 1
        low_freq_signals[pb_series > upper_band] = -1

        # Align to daily frequency
        daily_signals = pd.Series(0, index=data.index)
        aligned_signals = low_freq_signals.reindex(data.index).fillna(0)
        daily_signals.update(aligned_signals)

        # Store for plotting
        self.signals = daily_signals
        self.rolling_mean = rolling_mean_filled
        self.upper_band = upper_band
        self.lower_band = lower_band

        return daily_signals

    def plot_signals(self, data: pd.DataFrame):
        """
        Plot the P/B ratio, rolling mean, bands, and signals.

        Args:
            data: DataFrame containing 'Close' prices with daily frequency
        """
        if self.signals is None:
            raise ValueError("Signals have not been generated. Call generate_signals() first.")

        # Create single plot
        fig, ax = plt.subplots(figsize=(14, 8))

        # Use the resampled P/B data
        resampled_pb = self._resample_data(self.pb_series)

        ax.plot(resampled_pb.index, resampled_pb['PB_Ratio'],
                label='P/B Ratio', color='black', alpha=0.6)
        ax.plot(self.rolling_mean.index, self.rolling_mean,
                label='Rolling Mean', linestyle='--')
        ax.plot(self.upper_band.index, self.upper_band,
                label='Upper Band', linestyle=':')
        ax.plot(self.lower_band.index, self.lower_band,
                label='Lower Band', linestyle=':')

        # For the P/B plot, filter signals to match resampled frequency
        resampled_signals = self.signals.reindex(resampled_pb.index)
        buy_signals_pb = resampled_signals[resampled_signals == 1]
        sell_signals_pb = resampled_signals[resampled_signals == -1]

        if not buy_signals_pb.empty:
            ax.scatter(buy_signals_pb.index, resampled_pb.loc[buy_signals_pb.index, 'PB_Ratio'],
                       label='Buy Signal', marker='^', color='green', s=100)
        if not sell_signals_pb.empty:
            ax.scatter(sell_signals_pb.index, resampled_pb.loc[sell_signals_pb.index, 'PB_Ratio'],
                       label='Sell Signal', marker='v', color='red', s=100)

        ax.set_title(f'P/B Ratio with Bands ({self.frequency} Frequency)')
        ax.set_ylabel('P/B Ratio')
        ax.set_xlabel('Date')
        ax.legend()
        ax.grid(True)

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    import sys

    sys.path.append('..')  # Add parent directory to path
    from utils.backtest import Backtest
    from utils.data_handler import DataHandler

    data_handler = DataHandler(ticker='AAPL', start_date='2010-12-12', end_date='2016-12-12')
    prices = data_handler.fetch_data()

    # Assume pb_series is a pandas Series with P/B ratios indexed by date
    pb_series = data_handler.fetch_pb_series('../data/price_to_book_ratio.csv')

    # Test frequencies
    frequencies = ['weekly', 'monthly', '6month', 'yearly']
    results = {}

    for freq in frequencies:
        print(f"\nTesting {freq} frequency strategy:")

        # Initialize strategy with current frequency
        strategy = PriceToBookStrategy(pb_series, window=5, k=0.5, frequency=freq)

        # Run backtest
        backtest = Backtest(
            data=prices,
            strategy=strategy,
        )

        # Store results
        results[freq] = backtest.run()

        # Plot signals
        strategy.plot_signals(prices)

        # Print metrics
        print(f"\nResults for {freq} frequency:")
        print(f"Final Portfolio Value: ${results[freq]['Final Value']:,.2f}")
        print(f"Total Return: {results[freq]['Return']:.2%}")
        print(f"Sharpe Ratio: {results[freq]['Sharpe Ratio']:.2f}")
        print(f"Max Drawdown: {results[freq]['Max Drawdown']:.2%}")

    # Compare strategies
    print("\nStrategy Comparison:")
    comparison = pd.DataFrame({
        freq: {
            'Final Value': results[freq]['Final Value'],
            'Total Return': results[freq]['Return'],
            'Sharpe Ratio': results[freq]['Sharpe Ratio'],
            'Max Drawdown': results[freq]['Max Drawdown']
        }
        for freq in frequencies
    }).round(4)

    print(comparison)