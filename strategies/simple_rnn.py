import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Dropout
from tensorflow.keras.optimizers import Adam
from strategies.strategy import Strategy

class SimpleRNNStrategy(Strategy):
    def __init__(self, lookback=30, n_splits=5, epochs=30):
        self.lookback = lookback
        self.n_splits = n_splits
        self.epochs = epochs
        self.model = None
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.signals = None
        self.predictions = None
        self.cv_scores = []
        self.trained = False  # New flag to track training status

    def prepare_data(self, data):
        close_prices = data['Close'].values.reshape(-1, 1)
        scaled_prices = self.scaler_X.transform(close_prices)

        X, y = [], []
        for i in range(len(scaled_prices) - self.lookback):
            X.append(scaled_prices[i:(i + self.lookback)])
            y.append(scaled_prices[i + self.lookback])

        return np.array(X), np.array(y)

    def build_model(self):
        model = Sequential([
            SimpleRNN(50, activation='tanh', input_shape=(self.lookback, 1)),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model

    def cross_validate_and_train(self, X, y):
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        self.cv_scores = []

        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model = self.build_model()
            model.fit(X_train, y_train, epochs=self.epochs, batch_size=32,
                      validation_data=(X_val, y_val), verbose=0)

            val_score = model.evaluate(X_val, y_val, verbose=0)
            self.cv_scores.append(val_score)

        self.model = self.build_model()
        self.model.fit(X, y, epochs=self.epochs, batch_size=32, verbose=0)
        self.trained = True  # Mark as trained

    def fit(self, data):
        close_prices = data['Close'].values.reshape(-1, 1)
        self.scaler_X.fit(close_prices)
        self.scaler_y.fit(close_prices)

        X, y = self.prepare_data(data)
        self.cross_validate_and_train(X, y)

    def generate_prediction(self, X_new):
        return self.model.predict(X_new, verbose=0)

    def generate_signals(self, data):
        if not self.trained:
            self.fit(data)
        else:
            # Only re-fit scalers to match new data shape
            close_prices = data['Close'].values.reshape(-1, 1)
            self.scaler_X.fit(close_prices)
            self.scaler_y.fit(close_prices)

        X, y = self.prepare_data(data)
        predictions = self.generate_prediction(X)
        predictions = self.scaler_X.inverse_transform(predictions)

        prediction_index = data.index[self.lookback:]
        actual_prices = data['Close'].loc[prediction_index]
        pred_prices = pd.Series(predictions.flatten(), index=prediction_index)

        signals = pd.Series(0, index=data.index)
        signals[prediction_index] = np.where(pred_prices > actual_prices, 1, -1)

        self.signals = signals
        return signals

    def plot_signals(self, data):
        # Implementation for plotting signals
        pass
