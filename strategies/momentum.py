import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from strategies.strategy import Strategy


class MomentumStrategy(Strategy):
    """
    Class that aims to generate signals using an RSI-based momentum strategy.
    Allows setting custom overbought and oversold thresholds.
    """
    def __init__(self, period=14, overbought=70, oversold=30):
        self.period = period
        self.overbought = overbought
        self.oversold = oversold
        self.signals = None

    def set_signals(self, signals):
        self.signals = signals
        return

    def get_signals(self):
        return self.signals

    def generate_signals(self, data):
        signals, rsi = self.compute_rsi_signal(data, self.period, self.overbought, self.oversold)
        self.set_signals(signals)
        data['RSI'] = rsi  # Store RSI values for plotting
        return signals

    def compute_rsi_signal(self, data, period=14, overbought=70, oversold=30):
        delta = data["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        signals = (rsi < oversold).astype(int) - (rsi > overbought).astype(int)  # Buy: 1, Sell: -1
        return signals, rsi  # Buy: 1, Sell: -1

    def plot_signals(self, data):
        fig, ax1 = plt.subplots(figsize=(12, 7))  # Increased figure size for better readability
        ax1.plot(data['Close'], label='Close Price', color='black', linewidth=2, zorder=3)

        signals = self.get_signals()
        signal_changes = np.insert(np.diff(signals) != 0, 0, True)  # Always include the first signal

        buy_signals = data.loc[(signals == 1) & signal_changes]
        sell_signals = data.loc[(signals == -1) & signal_changes]

        ax1.scatter(buy_signals.index, buy_signals['Close'], marker='^', color='green', label='Buy Signal', s=80,
                    alpha=1, zorder=2)
        ax1.scatter(sell_signals.index, sell_signals['Close'], marker='v', color='red', label='Sell Signal', s=100,
                    alpha=1, zorder=2)
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.set_title('RSI Trading Signals')

        ax2 = ax1.twinx()
        ax2.plot(data['RSI'], label='RSI', color='cornflowerblue', linestyle='solid', linewidth=1, zorder=0)
        ax2.axhline(y=self.overbought, color='red', linestyle='dotted', label='Overbought')
        ax2.axhline(y=self.oversold, color='green', linestyle='dotted', label='Oversold')
        ax2.set_ylabel('RSI')
        ax2.legend(loc='upper right')  # Send RSI plot to the background

        plt.xlabel('Date')
        plt.tight_layout()
        plt.show()
