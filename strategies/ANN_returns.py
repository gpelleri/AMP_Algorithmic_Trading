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

class ANNReturnStrategy(Strategy):
    def __init__(self, epochs=50, batch_size=32):
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
        
        # Momentum (plus important pour les rendements)
        df['RSI'] = RSIIndicator(df['Close']).rsi()
        df['RSI_Change'] = df['RSI'].diff()  # Variation du RSI
        df['MACD'] = MACD(df['Close']).macd_diff()
        
        # Tendance relative
        df['SMA_cross'] = SMAIndicator(df['Close'], window=20).sma_indicator() / \
                         SMAIndicator(df['Close'], window=50).sma_indicator() - 1
        
        # Volatilité
        df['BB_width'] = BollingerBands(df['Close']).bollinger_wband()
        
        df.dropna(inplace=True)
        return df

    def generate_signals(self, df):
        original_index = df.index  # Sauvegarder l'index original
        df = self.compute_indicators(df)
        features = ['RSI', 'RSI_Change', 'MACD', 'SMA_cross', 'BB_width']
        X = self.scaler.fit_transform(df[features])
        y = df['Return'].values

        # Train/test split tout en gardant l'ordre temporel
        test_size = int(len(df) * 0.2)
        train_size = len(df) - test_size
        X_train = X[:train_size]
        X_test = X[train_size:]
        y_train = y[:train_size]
        y_test = y[train_size:]

        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(X.shape[1],)),
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

        # Placer les signaux dans la période de test uniquement
        signals.iloc[-test_size:] = test_signals

        self.signals = signals
        return signals