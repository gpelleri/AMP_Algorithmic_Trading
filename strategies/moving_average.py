import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from strategies.strategy import Strategy


class MovingAverage(Strategy):
    """
    Simple moving average Strategy class. Allows to use both 'crossover' MA and 'single' MA with the "mode" parameter.
    short_window & long_window are the window on which we will compute the moving average. We only use short_window
    when using 'single' mode.
    By default, we BUY if short avg > long avg & SELL if short_avg > long_avg.
    'invert_signals' is a flag so switch the behavior of buy & sells signals.
    """
    def __init__(self, short_window=20, long_window=50, invert_signals=False, mode="crossover"):
        self.short_window = short_window
        self.long_window = long_window
        self.invert_signals = invert_signals
        self.mode = mode  # 'crossover' or 'single'
        self.signals = None

    def set_signals(self, signals):
        self.signals = signals
        return

    def get_signals(self):
        return self.signals

    def generate_signals(self, data):
        short_ma = data['Close'].rolling(window=self.short_window).mean()
        long_ma = data['Close'].rolling(window=self.long_window).mean()

        if self.mode == 'crossover':
            signals = np.where(short_ma > long_ma, 1, np.where(short_ma < long_ma, -1, 0))
        else:  # 'single' mode
            signals = np.where(data['Close'] > short_ma, 1, np.where(data['Close'] < short_ma, -1, 0))

        if self.invert_signals:
            signals *= -1

        self.set_signals(signals)

        return pd.Series(signals, index=data.index)

    def plot_signals(self, data):
        short_ma = data['Close'].rolling(window=self.short_window).mean()
        long_ma = data['Close'].rolling(window=self.long_window).mean() if self.mode == 'crossover' else None

        plt.figure(figsize=(12, 6))
        plt.plot(data['Close'], label='Close Price', color='black')
        plt.plot(short_ma, label=f'Short {self.short_window}-day MA', color='blue')
        if long_ma is not None:
            plt.plot(long_ma, label=f'Long {self.long_window}-day MA', color='red')

        signals = self.get_signals()

        # Identify where the signal changes
        signal_changes = np.insert(np.diff(signals) != 0, 0, True)  # Always include the first signal

        buy_signals = data.loc[(signals == 1) & signal_changes]
        sell_signals = data.loc[(signals == -1) & signal_changes]

        plt.scatter(buy_signals.index, buy_signals['Close'], marker='^', color='green', label='Buy Signal', s=150, alpha=1)
        plt.scatter(sell_signals.index, sell_signals['Close'], marker='v', color='red', label='Sell Signal', s=200, alpha=1)

        plt.legend()
        plt.title('Trading Signals')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.show()

