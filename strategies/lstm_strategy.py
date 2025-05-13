import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
from strategies.strategy import Strategy

class LSTMStrategy(Strategy):
    """
    LSTM Strategy for multi-period price or return prediction.
    """
    def __init__(self, lookback=30, n_splits=5, epochs=30, n_days=1, predict_returns=False):
        self.lookback = lookback
        self.n_splits = n_splits
        self.epochs = epochs
        self.n_days = n_days
        self.predict_returns = predict_returns

        self.model = None
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.signals = None
        self.predictions = None
        self.cv_scores = []

    def prepare_data(self, data):
        close_prices = data['Close'].values.reshape(-1, 1)

        # Compute returns if required
        if self.predict_returns:
            returns = np.diff(close_prices, axis=0) / close_prices[:-1]
            close_prices = returns
            close_prices = close_prices.reshape(-1, 1)

        scaled_prices = self.scaler_X.fit_transform(close_prices)

        X, y = [], []

        for i in range(len(scaled_prices) - self.lookback - self.n_days + 1):
            X.append(scaled_prices[i:(i + self.lookback)])
            y.append(scaled_prices[i + self.lookback:i + self.lookback + self.n_days].flatten())

        X = np.array(X)
        y = np.array(y)

        return X, y

    def build_model(self):
        model = Sequential([
            LSTM(50, activation='tanh', input_shape=(self.lookback, 1)),
            Dropout(0.2),
            Dense(self.n_days)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model

    def cross_validate_and_train(self, X, y):
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        cv_scores = []

        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model = self.build_model()
            model.fit(X_train, y_train, epochs=self.epochs, batch_size=32,
                      validation_data=(X_val, y_val), verbose=0)

            val_score = model.evaluate(X_val, y_val, verbose=0)
            cv_scores.append(val_score)

        self.cv_scores = cv_scores
        self.model = self.build_model()
        self.model.fit(X, y, epochs=self.epochs, batch_size=32, verbose=0)

    def generate_prediction(self, X_new):
        return self.model.predict(X_new, verbose=0)

    def generate_signals(self, data):
        X, y = self.prepare_data(data)

        if self.model is None:
            self.cross_validate_and_train(X, y)

        preds = self.generate_prediction(X)

        if self.predict_returns:
            preds = self.scaler_X.inverse_transform(preds)

            # Aligner correctement les prix de base pour reconstruire les futurs
            base_index_start = self.lookback - 1
            base_index_end = base_index_start + len(preds)
            last_known = data['Close'].values[base_index_start:base_index_end]

            future_prices = last_known.reshape(-1, 1) * (1 + preds[:, 0].reshape(-1, 1))
        else:
            preds = self.scaler_X.inverse_transform(preds)
            future_prices = preds[:, 0].reshape(-1, 1)

        # Aligner l'index des prédictions
        prediction_index = data.index[self.lookback + self.n_days - 1:self.lookback + self.n_days - 1 + len(future_prices)]

        actual_prices = data['Close'].loc[prediction_index]
        pred_series = pd.Series(future_prices.flatten(), index=prediction_index)

        signals = pd.Series(0, index=data.index)
        signals[prediction_index] = np.where(pred_series > actual_prices, 1, -1)

        self.signals = signals
        return signals

    def plot_signals(self, data):
        # Placeholder for plotting
        pass
