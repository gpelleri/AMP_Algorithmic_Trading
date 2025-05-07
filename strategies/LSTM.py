import numpy as np
import pandas as pd
import ta
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from strategies.strategy import Strategy
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.volatility import BollingerBands

class LSTMStrategy(Strategy):
    def __init__(self, window_size=20, epochs=20, batch_size=32):
        self.window_size = window_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.scaler = StandardScaler()
        self.signals = None

    def get_signals(self):
        return self.signals

    def compute_indicators(self, df):
        df = df.copy()
        df['Return'] = df['Close'].pct_change().shift(-1)
        df['RSI'] = RSIIndicator(df['Close']).rsi()
        df['MACD'] = MACD(df['Close']).macd_diff()
        df['BB_width'] = BollingerBands(df['Close']).bollinger_wband()
        df.dropna(inplace=True)
        return df

    def prepare_sequences(self, data, target_column='Return'):
        X, y = [], []
        for i in range(self.window_size, len(data)):
            X.append(data[i - self.window_size:i])
            y.append(data[i][target_column_index])
        return np.array(X), np.array(y)

    def generate_signals(self, df):
        original_index = df.index  # Sauvegarder l'index original
        df = self.compute_indicators(df)
        features = ['RSI', 'MACD', 'BB_width']
        df_features = self.scaler.fit_transform(df[features])
        X, y = self.create_dataset(df_features, df['Return'].values)

        # Train/test split tout en gardant l'ordre temporel
        test_size = int(len(X) * 0.2)
        X_train = X[:-test_size]
        X_test = X[-test_size:]
        y_train = y[:-test_size]
        y_test = y[-test_size:]

        self.model = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(50, return_sequences=False, input_shape=(self.window_size, X.shape[2])),
            tf.keras.layers.Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mse')
        self.model.fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=0)

        # Créer une série avec l'index original complet
        signals = pd.Series(0, index=original_index)

        # Faire les prédictions sur les données de test
        y_pred = self.model.predict(X_test)

        # Convertir les prédictions en signaux (-1, 0, 1)
        test_signals = (y_pred.flatten() > 0).astype(int) - (y_pred.flatten() < 0).astype(int)

        # Nous devons ajuster l'indice pour tenir compte de la fenêtre glissante
        test_start_idx = len(original_index) - len(test_signals)
        signals.iloc[test_start_idx:] = test_signals

        self.signals = signals
        return signals

    def create_dataset(self, X_data, y_data):
        X_seq, y_seq = [], []
        for i in range(self.window_size, len(X_data)):
            X_seq.append(X_data[i - self.window_size:i])
            y_seq.append(y_data[i])
        return np.array(X_seq), np.array(y_seq)