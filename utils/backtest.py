import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from strategies.strategy import Strategy


class Backtest:
    def __init__(self, data: pd.DataFrame, strategy: Strategy, initial_cash=10000, transaction_cost=0.001):
        self.data = data
        self.strategy = strategy
        self.initial_cash = initial_cash
        self.transaction_cost = transaction_cost
        self.positions = pd.DataFrame(index=data.index, columns=['position'])
        self.cash = initial_cash
        self.holdings = 0
        self.trades = []

    def run(self):
        signals = self.strategy.generate_signals(self.data)

        for i in range(len(self.data)):
            signal = signals.iloc[i]
            price = self.data['Close'].iloc[i]

            if signal == 1:
                self.buy(price)
            elif signal == -1:
                self.sell(price)

            self.positions.iloc[i] = self.holdings

        return self.calculate_metrics()

    def buy(self, price):
        if self.cash > 0:
            units = self.cash / price
            cost = units * price * (1 + self.transaction_cost)
            if cost <= self.cash:
                self.holdings += units
                self.cash -= cost
                self.trades.append(('BUY', price, units))

    def sell(self, price):
        if self.holdings > 0: # We don't autorise short selling by default
            revenue = self.holdings * price * (1 - self.transaction_cost)
            self.cash += revenue
            self.trades.append(('SELL', price, self.holdings))
            self.holdings = 0

    def calculate_metrics(self):
        portfolio_value = self.cash + (self.holdings * self.data['Close'])
        returns = portfolio_value.pct_change().dropna()
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
        drawdown = (portfolio_value / portfolio_value.cummax() - 1).min()

        self.visualize_results(portfolio_value)

        return {
            'Final Value': portfolio_value.iloc[-1],
            'Return': portfolio_value.iloc[-1] / self.initial_cash - 1,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown': drawdown
        }

    def visualize_results(self, portfolio_value):
        plt.figure(figsize=(10, 5))
        plt.plot(portfolio_value, label='Portfolio Value')
        plt.legend()
        plt.title('Backtest Performance')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value')
        plt.show()

    def plot_signals(self):
        short_ma = self.data['Close'].rolling(window=self.strategy.short_window).mean()
        long_ma = self.data['Close'].rolling(
            window=self.strategy.long_window).mean() if self.strategy.mode == 'crossover' else None

        plt.figure(figsize=(12, 6))
        plt.plot(self.data['Close'], label='Close Price', color='black')
        plt.plot(short_ma, label=f'Short {self.strategy.short_window}-day MA', color='blue')
        if long_ma is not None:
            plt.plot(long_ma, label=f'Long {self.strategy.long_window}-day MA', color='red')

        signals = self.strategy.generate_signals(self.data)
        buy_signals = self.data.loc[signals == 1]
        sell_signals = self.data.loc[signals == -1]

        plt.scatter(buy_signals.index, buy_signals['Close'], marker='^', color='green', label='Buy Signal', alpha=1)
        plt.scatter(sell_signals.index, sell_signals['Close'], marker='v', color='red', label='Sell Signal', alpha=1)

        plt.legend()
        plt.title('Trading Signals')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.show()
