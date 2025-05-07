import numpy as np
import pandas as pd
import ta
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from strategies.strategy import Strategy
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.volatility import BollingerBands


class CNNStrategy(Strategy):
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
        # Rendement cible
        df['Return'] = df['Close'].pct_change().shift(-1)
        
        # Momentum
        df['RSI'] = RSIIndicator(df['Close']).rsi()
        df['RSI_Change'] = df['RSI'].diff()
        df['MACD'] = MACD(df['Close']).macd_diff()
        
        # Tendance
        df['SMA20'] = SMAIndicator(df['Close'], window=20).sma_indicator()
        df['SMA50'] = SMAIndicator(df['Close'], window=50).sma_indicator()
        df['Price_SMA20'] = df['Close'] / df['SMA20'] - 1
        
        # Volatilité
        bb = BollingerBands(df['Close'])
        df['BB_width'] = bb.bollinger_wband()
        df['BB_position'] = (df['Close'] - bb.bollinger_lband()) / \
                           (bb.bollinger_hband() - bb.bollinger_lband())
        
        df.dropna(inplace=True)
        return df

    def generate_signals(self, df):
        original_index = df.index  # Sauvegarder l'index original
        df = self.compute_indicators(df)
        features = ['RSI', 'RSI_Change', 'MACD', 'SMA20', 'SMA50', 'Price_SMA20', 
                   'BB_width', 'BB_position']
        df_features = self.scaler.fit_transform(df[features])
        X, y = self.create_dataset(df_features, df['Return'].values)

        # Train/test split tout en gardant l'ordre temporel
        test_size = int(len(X) * 0.2)
        X_train = X[:-test_size]
        X_test = X[-test_size:]
        y_train = y[:-test_size]
        y_test = y[-test_size:]

        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu',
                                   input_shape=(self.window_size, X.shape[2])),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(32, activation='relu'),
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