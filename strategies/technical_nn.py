import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from strategies.strategy import Strategy
import matplotlib.pyplot as plt

class TechnicalNNStrategy(Strategy):
    """
    Neural Network Strategy using technical indicators for S&P 500 stocks.
    Supports both price and return predictions with out-of-sample evaluation.
    """
    def __init__(self, prediction_type='price', n_splits=5, epochs=50):
        self.prediction_type = prediction_type  # 'price' or 'return'
        self.n_splits = n_splits
        self.epochs = epochs
        self.model = None
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.signals = None
        self.predictions = None
        self.cv_scores = []
        self.trained = False  # Flag for out-of-sample capability

    def calculate_technical_indicators(self, data):
        df = data.copy()

        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # Moving averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()

        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

        # Bollinger Bands
        df['BB_middle'] = df['Close'].rolling(window=20).mean()
        df['BB_upper'] = df['BB_middle'] + 2 * df['Close'].rolling(window=20).std()
        df['BB_lower'] = df['BB_middle'] - 2 * df['Close'].rolling(window=20).std()

        return df.fillna(method='bfill')

    def prepare_data(self, data, fit_scalers=False):
        df = self.calculate_technical_indicators(data)
        features = ['RSI', 'SMA_20', 'SMA_50', 'MACD', 'Signal_Line',
                    'BB_middle', 'BB_upper', 'BB_lower']

        X = df[features].values

        if self.prediction_type == 'return':
            y = df['Close'].pct_change().shift(-1).values
        else:
            y = df['Close'].values

        # Remove NaNs
        mask = ~np.isnan(y)
        X = X[mask]
        y = y[mask]

        if fit_scalers:
            X = self.scaler_X.fit_transform(X)
            y = self.scaler_y.fit_transform(y.reshape(-1, 1))
        else:
            X = self.scaler_X.transform(X)
            y = self.scaler_y.transform(y.reshape(-1, 1))

        return X, y

    def build_model(self, input_dim):
        model = Sequential([
            Dense(64, activation='relu', input_dim=input_dim),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.1),
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

            model = self.build_model(X.shape[1])
            model.fit(X_train, y_train, epochs=self.epochs, batch_size=32,
                      validation_data=(X_val, y_val), verbose=0)

            val_score = model.evaluate(X_val, y_val, verbose=0)
            self.cv_scores.append(val_score)

        self.model = self.build_model(X.shape[1])
        self.model.fit(X, y, epochs=self.epochs, batch_size=32, verbose=0)
        self.trained = True  # Out-of-sample capable

    def fit(self, data):
        X, y = self.prepare_data(data, fit_scalers=True)
        self.cross_validate_and_train(X, y)

    def generate_prediction(self, X):
        return self.model.predict(X, verbose=0)

    def generate_signals(self, data):
        if not self.trained:
            self.fit(data)
        else:
            # For out-of-sample prediction, update features only
            X, y = self.prepare_data(data, fit_scalers=False)

        predictions = self.generate_prediction(X)
        predictions = self.scaler_y.inverse_transform(predictions)

        if self.prediction_type == 'return':
            base_signals = np.where(predictions > 0, 1, -1).flatten()
            signals = np.zeros(len(base_signals))
            signals[:-1] = base_signals[1:]
        else:  # 'price'
            signals = np.zeros(len(predictions))
            signals[1:] = np.where(predictions[1:] > predictions[:-1], 1, -1).flatten()

        self.predictions = predictions
        signals_partial = pd.Series(signals, index=data.index[-len(signals):])

        # Reindex to full data length, fill missing with 0
        full_signals = pd.Series(0, index=data.index)
        full_signals.update(signals_partial)

        self.signals = signals_partial  # keep original for analysis if needed
        return full_signals

    def plot_signals(self, data):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

        # Plot 1: Price/Returns and Predictions
        if self.prediction_type == 'return':
            actual = data['Close'].pct_change()
            ax1.plot(actual.index[-len(self.predictions):], actual[-len(self.predictions):],
                    label='Actual Returns', color='blue')
            ax1.plot(actual.index[-len(self.predictions):], self.predictions,
                    label='Predicted Returns', color='red', linestyle='--')
            ax1.set_title('Returns Prediction')
        else:
            ax1.plot(data.index[-len(self.predictions):], data['Close'][-len(self.predictions):],
                    label='Actual Price', color='blue')
            ax1.plot(data.index[-len(self.predictions):], self.predictions,
                    label='Predicted Price', color='red', linestyle='--')
            ax1.set_title('Price Prediction')

        ax1.legend()
        ax1.grid(True)

        # Plot 2: Trading Signals
        ax2.plot(self.signals.index, self.signals, label='Trading Signals', color='green')
        ax2.set_title('Generated Trading Signals')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.show()

        # Print cross-validation scores
        print("\nCross-validation MSE scores:")
        for i, score in enumerate(self.cv_scores, 1):
            print(f"Fold {i}: {score:.6f}")
        print(f"Average MSE: {np.mean(self.cv_scores):.6f} (±{np.std(self.cv_scores):.6f})")
