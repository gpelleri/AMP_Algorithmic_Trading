import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from strategies.strategy import Strategy


class RatioValueStrategy(Strategy):
    """
    A trading strategy based on value ratio (P/E, P/B etc).
    Generates signals based on statistical bands around the rolling mean of the ratio.
    """

    FREQ_MAPPING = {
        'weekly': 'W-FRI',  # Weekly data, Friday as end of week
        'monthly': 'M',  # Monthly data, last day of month
        '6month': '6M',  # 6-month data, last day of 6-month period
        'yearly': 'Y',  # Yearly data, last day of year
    }

    def __init__(self, ratio_series, ratio, window: int = 5, k: float = 0.5, frequency: str = 'weekly', ):
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

        if ratio not in ["PB", "PE"]:
            raise ValueError("Implemented Ratio are PB and PE. Do not put any / or space in the ratio name.")

        self.ratio_series = ratio_series
        self.frequency = frequency
        self.ratio = ratio+"_Ratio"
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
        Generate trading signals based on ratio.

        Args:
            data: DataFrame containing 'Close' and 'PB_Ratio' columns.
                 Will be resampled according to the strategy's frequency.

        Returns:
            Series of trading signals (-1, 0, 1)
        """
        # Resample the input data
        resampled_pb = self._resample_data(self.ratio_series)
        ratio_series = resampled_pb[self.ratio]

        # Calculate rolling statistics
        rolling_mean = ratio_series.rolling(window=self.window).mean()
        rolling_std = ratio_series.rolling(window=self.window).std()

        # Fill missing values using forward-fill and backward-fill (no look-ahead)
        rolling_mean_filled = rolling_mean.ffill().bfill()
        rolling_std_filled = rolling_std.ffill().bfill()

        # Calculate upper and lower bands
        upper_band = rolling_mean_filled + self.k * rolling_std_filled
        lower_band = rolling_mean_filled - self.k * rolling_std_filled

        # Generate low-frequency signals (e.g., weekly/monthly)
        low_freq_signals = pd.Series(0, index=ratio_series.index)
        low_freq_signals[ratio_series < lower_band] = 1
        low_freq_signals[ratio_series > upper_band] = -1

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

        # Use the resampled  data
        resampled = self._resample_data(self.ratio_series)

        ax.plot(resampled.index, resampled[self.ratio],
                label=f'{self.ratio}', color='black', alpha=0.6)
        ax.plot(self.rolling_mean.index, self.rolling_mean,
                label='Rolling Mean', linestyle='--')
        ax.plot(self.upper_band.index, self.upper_band,
                label='Upper Band', linestyle=':')
        ax.plot(self.lower_band.index, self.lower_band,
                label='Lower Band', linestyle=':')

        # For the ratio plot, filter signals to match resampled frequency
        resampled_signals = self.signals.reindex(resampled.index)
        buy_signals = resampled_signals[resampled_signals == 1]
        sell_signals = resampled_signals[resampled_signals == -1]

        if not buy_signals.empty:
            ax.scatter(buy_signals.index, resampled.loc[buy_signals.index, self.ratio],
                       label='Buy Signal', marker='^', color='green', s=100)
        if not sell_signals.empty:
            ax.scatter(sell_signals.index, resampled.loc[sell_signals.index, self.ratio],
                       label='Sell Signal', marker='v', color='red', s=100)

        ax.set_title(f'{self.ratio}with Bands ({self.frequency} Frequency)')
        ax.set_ylabel(f'{self.ratio}')
        ax.set_xlabel('Date')
        ax.legend()
        ax.grid(True)

        plt.tight_layout()
        plt.show()


