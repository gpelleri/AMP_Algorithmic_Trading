import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from strategies.strategy import Strategy


class MomentumStrategy(Strategy):
    """
    Placeholder class for now.
    """
    def __init__(self, lookback=14):
        self.lookback = lookback
        self.signals = None

    def set_signals(self, signals):
        self.signals = signals
        return

    def get_signals(self):
        return self.signals

    def generate_signals(self, data):
        momentum = data['Close'].diff(self.lookback)
        signals = np.where(momentum > 0, 1, np.where(momentum < 0, -1, 0))

        self.set_signals(signals)
        return pd.Series(signals, index=data.index)

    def plot_signals(self, data):
        plt.figure(figsize=(12, 6))
        plt.plot(data['Close'], label='Close Price', color='black')

        signals = self.generate_signals(data)
        buy_signals = data.loc[signals == 1]
        sell_signals = data.loc[signals == -1]

        plt.scatter(buy_signals.index, buy_signals['Close'], marker='^', color='green', label='Buy Signal', alpha=1)
        plt.scatter(sell_signals.index, sell_signals['Close'], marker='v', color='red', label='Sell Signal', alpha=1)

        plt.legend()
        plt.title('Momentum Trading Signals')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.show()