import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import floor
from strategies.strategy import Strategy


class Backtest:
    def __init__(self, data: pd.DataFrame, strategy: Strategy,
                 benchmark_data: pd.DataFrame = None,
                 initial_cash=100000.0,
                 transaction_cost=0.01,
                 risk_free_rate=0.0,
                 plot_results=True):
        self.data = data
        self.strategy = strategy
        self.benchmark_data = benchmark_data
        self.initial_cash = initial_cash
        self.transaction_cost = transaction_cost
        self.risk_free_rate = risk_free_rate  # Annualized risk-free rate

        self.portfolio = pd.DataFrame(index=data.index, columns=['cash', 'holdings'])
        self.portfolio['cash'] = initial_cash
        self.portfolio['holdings'] = 0
        self.trades = []
        self.plot = plot_results

    def run(self):
        signals = self.strategy.generate_signals(self.data).shift(1)

        for i in range(1, len(signals)):
            signal = signals.iloc[i]
            price = self.data["Close"].iloc[i]

            self.portfolio.at[self.data.index[i], 'cash'] = self.portfolio.at[self.data.index[i - 1], 'cash']
            self.portfolio.at[self.data.index[i], 'holdings'] = self.portfolio.at[self.data.index[i - 1], 'holdings']

            if signal == 1:
                self.buy(i, price)
            elif signal == -1:
                self.sell(i, price)

        return self.calculate_metrics()

    def buy(self, i, price):
        cash_available = self.portfolio.at[self.data.index[i - 1], 'cash']
        if cash_available > 0:
            units = floor(cash_available / (price * (1 + self.transaction_cost)))
            cost = units * price * (1 + self.transaction_cost)
            if cost <= cash_available and units > 0:
                self.portfolio.at[self.data.index[i], 'holdings'] += units
                self.portfolio.at[self.data.index[i], 'cash'] -= cost
                self.trades.append(('BUY', price, units))

    def sell(self, i, price):
        holdings_available = self.portfolio.at[self.data.index[i], 'holdings']
        if holdings_available > 0:
            revenue = holdings_available * price * (1 - self.transaction_cost)
            self.portfolio.at[self.data.index[i], 'cash'] += revenue
            self.portfolio.at[self.data.index[i], 'holdings'] = 0
            self.trades.append(('SELL', price, holdings_available))

    def calculate_metrics(self):
        self.portfolio['portfolio_value'] = (
            self.portfolio['cash'] + self.portfolio['holdings'] * self.data["Close"]
        )

        strategy_returns = self.portfolio['portfolio_value'].pct_change().dropna()
        excess_strategy_returns = strategy_returns - self.risk_free_rate / 252
        strategy_sharpe = (
            np.mean(excess_strategy_returns) / np.std(strategy_returns) * np.sqrt(252)
            if np.std(strategy_returns) > 0 else np.nan
        )
        strategy_drawdown = (
            self.portfolio['portfolio_value'] / self.portfolio['portfolio_value'].cummax() - 1
        ).min()

        benchmark_metrics = {}
        if self.benchmark_data is not None:
            benchmark_close = self.benchmark_data['Close'].reindex(self.data.index).fillna(method='ffill')
            benchmark_returns = benchmark_close.pct_change().dropna()
            excess_benchmark_returns = benchmark_returns - self.risk_free_rate / 252

            benchmark_cum_return = (1 + benchmark_returns).cumprod()
            benchmark_sharpe = (
                np.mean(excess_benchmark_returns) / np.std(benchmark_returns) * np.sqrt(252)
                if np.std(benchmark_returns) > 0 else np.nan
            )
            benchmark_drawdown = (
                benchmark_cum_return / benchmark_cum_return.cummax() - 1
            ).min()

            benchmark_metrics = {
                'Benchmark Return': benchmark_cum_return.iloc[-1] - 1,
                'Benchmark Sharpe Ratio': benchmark_sharpe,
                'Benchmark Max Drawdown': benchmark_drawdown
            }

        if self.plot:
            self.visualize_results()

        return {
            'Final Value': self.portfolio['portfolio_value'].iloc[-1],
            'Return': self.portfolio['portfolio_value'].iloc[-1] / self.initial_cash - 1,
            'Sharpe Ratio': strategy_sharpe,
            'Max Drawdown': strategy_drawdown,
            **benchmark_metrics
        }

    def visualize_results(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.portfolio['portfolio_value'], label='Strategy Portfolio', color='blue')

        if self.benchmark_data is not None:
            benchmark_close = self.benchmark_data['Close'].reindex(self.data.index).fillna(method='ffill')
            benchmark_cum_return = (1 + benchmark_close.pct_change().fillna(0)).cumprod() * self.initial_cash
            plt.plot(benchmark_cum_return, label='S&P 500 (Benchmark)', color='orange')

        plt.legend()
        plt.title('Strategy vs Benchmark Performance')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value')
        plt.grid(True)
        plt.tight_layout()
        plt.show()