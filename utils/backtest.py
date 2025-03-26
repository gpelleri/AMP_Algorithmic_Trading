import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import floor
from strategies.strategy import Strategy


class Backtest:
    def __init__(self, data: pd.DataFrame, strategy: Strategy, initial_cash=100000.0, transaction_cost=0.001):
        self.data = data
        self.strategy = strategy
        self.initial_cash = initial_cash
        self.transaction_cost = transaction_cost

        # Initialize a DataFrame to store cash and holdings over time
        self.portfolio = pd.DataFrame(index=data.index, columns=['cash', 'holdings'])
        self.portfolio['cash'] = initial_cash
        self.portfolio['holdings'] = 0

        self.trades = []

    def run(self):
        signals = self.strategy.generate_signals(self.data)

        for i in range(1, len(self.data)):
            signal = signals.iloc[i]
            price = self.data["Close"].iloc[i]

            # Carry forward previous values
            self.portfolio.at[self.data.index[i], 'cash'] = self.portfolio.at[self.data.index[i - 1], 'cash']
            self.portfolio.at[self.data.index[i], 'holdings'] = self.portfolio.at[self.data.index[i - 1], 'holdings']

            if signal == 1:
                self.buy(i, price)
            elif signal == -1:
                self.sell(i, price)

        return self.calculate_metrics()

    def buy(self, i, price):
        cash_available = self.portfolio.at[self.data.index[i-1], 'cash']
        if cash_available > 0:
            units = floor(cash_available / (price * (1 + self.transaction_cost)))
            cost = units * price * (1 + self.transaction_cost)
            if (cost <= cash_available) and (units > 0):
                self.portfolio.at[self.data.index[i], 'holdings'] += units
                self.portfolio.at[self.data.index[i], 'cash'] -= cost
                self.trades.append(('BUY', price, units))

    def sell(self, i, price):
        holdings_available = self.portfolio.at[self.data.index[i], 'holdings']
        # TODO : check shortselling & implement it
        cash_available = self.portfolio.at[self.data.index[i], 'cash']

        if holdings_available > 0:
            # Regular sell: sell all holdings
            revenue = holdings_available * price * (1 - self.transaction_cost)
            self.portfolio.at[self.data.index[i], 'cash'] += revenue
            self.portfolio.at[self.data.index[i], 'holdings'] = 0
            self.trades.append(('SELL', price, holdings_available))

    def calculate_metrics(self):
        # Compute portfolio value (cash + holdings * price)
        self.portfolio['portfolio_value'] = (
            self.portfolio['cash'] + (self.portfolio['holdings'] * self.data["Close"])
        )

        returns = self.portfolio['portfolio_value'].pct_change().dropna()
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if returns.std() != 0 else np.nan
        drawdown = (self.portfolio['portfolio_value'] / self.portfolio['portfolio_value'].cummax() - 1).min()

        self.visualize_results()

        return {
            'Final Value': self.portfolio['portfolio_value'].iloc[-1],
            'Return': self.portfolio['portfolio_value'].iloc[-1] / self.initial_cash - 1,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown': drawdown
        }

    def visualize_results(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.portfolio['portfolio_value'], label='Portfolio Value', color='blue')
        plt.legend()
        plt.title('Backtest Performance')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value')
        plt.show()
