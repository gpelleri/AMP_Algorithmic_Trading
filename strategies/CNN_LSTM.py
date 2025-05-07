import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from strategies.strategy import Strategy
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.volatility import BollingerBands

class HybridCNNLSTMStrategy(Strategy):
    """Architecture hybride CNN+LSTM"""

    def __init__(self, window_size=20, epochs=50, batch_size=32):
        self.window_size = window_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.signals = None

    def compute_indicators(self, df):
        df = df.copy()
        df['RSI'] = RSIIndicator(df['Close']).rsi()
        df['MACD'] = MACD(df['Close']).macd_diff()
        df['BB_width'] = BollingerBands(df['Close']).bollinger_wband()
        df['SMA_20'] = SMAIndicator(df['Close'], window=20).sma_indicator()
        df['EMA_20'] = EMAIndicator(df['Close'], window=20).ema_indicator()
        df['HL_PCT'] = (df['High'] - df['Low']) / df['Close']  # Volatilité High-Low
        df['Return'] = df['Close'].pct_change()
        df.dropna(inplace=True)
        return df

    def create_dataset(self, X_data, y_data):
        X, y = [], []
        for i in range(self.window_size, len(X_data)):
            X.append(X_data[i - self.window_size:i])
            y.append(y_data[i])
        return np.array(X), np.array(y)

    def generate_signals(self, df):
        original_index = df.index
        df = self.compute_indicators(df)

        features = ['RSI', 'MACD', 'BB_width', 'SMA_20', 'EMA_20', 'Return']
        X = self.scaler_X.fit_transform(df[features])
        y = self.scaler_y.fit_transform(df[['Close']])

        X_seq, y_seq = self.create_dataset(X, y)

        test_size = int(len(X_seq) * 0.2)
        X_train = X_seq[:-test_size]
        X_test = X_seq[-test_size:]
        y_train = y_seq[:-test_size]
        y_test = y_seq[-test_size:]

        # Architecture hybride CNN+LSTM
        self.model = tf.keras.models.Sequential([
            # Couches CNN pour l'extraction de features
            tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu',
                                   input_shape=(self.window_size, len(features))),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu'),
            tf.keras.layers.MaxPooling1D(pool_size=2),

            # Couches LSTM pour capturer les dépendances temporelles
            tf.keras.layers.LSTM(100, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(50),

            # Couches denses pour la prédiction finale
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mse')
        self.model.fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=0)

        signals = pd.Series(0, index=original_index)
        y_pred = self.model.predict(X_test)
        # Conversion en signaux
        test_signals = np.zeros(len(y_pred))
        y_pred_actual = self.scaler_y.inverse_transform(y_pred).flatten()  # Aplatir le tableau
        y_test_actual = self.scaler_y.inverse_transform(y_test).flatten()  # Aplatir le tableau
        test_signals[1:] = (y_pred_actual[1:] > y_pred_actual[:-1]).astype(int) - \
                           (y_pred_actual[1:] < y_pred_actual[:-1]).astype(int)

        test_start_idx = len(original_index) - len(test_signals)
        signals.iloc[test_start_idx:] = test_signals

        self.signals = signals
        return signals