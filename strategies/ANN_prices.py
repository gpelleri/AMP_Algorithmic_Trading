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

class ANNPriceStrategy(Strategy):
    """ANN pour la prédiction directe des prix"""

    def __init__(self, epochs=50, batch_size=32):
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.signals = None

    def compute_indicators(self, df):
        df = df.copy()
        # Prix cible
        df['Price_Next'] = df['Close'].shift(-1)

        # Tendance (niveaux absolus importants pour les prix)
        df['SMA20'] = SMAIndicator(df['Close'], window=20).sma_indicator()
        df['SMA50'] = SMAIndicator(df['Close'], window=50).sma_indicator()
        df['Price_SMA20'] = df['Close'] / df['SMA20'] - 1  # Distance au SMA

        # Momentum
        df['RSI'] = RSIIndicator(df['Close']).rsi()
        df['MACD'] = MACD(df['Close']).macd_diff()

        # Volatilité et support/résistance
        bb = BollingerBands(df['Close'])
        df['BB_upper'] = bb.bollinger_hband()
        df['BB_lower'] = bb.bollinger_lband()
        df['Price_BB_up'] = df['Close'] / df['BB_upper'] - 1  # Distance aux bandes

        df.dropna(inplace=True)
        return df

    def generate_signals(self, df):
        original_index = df.index
        df = self.compute_indicators(df)

        features = ['SMA20', 'SMA50', 'Price_SMA20', 'RSI', 'MACD',
                   'BB_upper', 'BB_lower', 'Price_BB_up']
        X = self.scaler_X.fit_transform(df[features])
        y = self.scaler_y.fit_transform(df[['Price_Next']])

        test_size = int(len(df) * 0.2)
        X_train = X[:-test_size]
        X_test = X[-test_size:]
        y_train = y[:-test_size]
        y_test = y[-test_size:]

        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(len(features),)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mse')
        self.model.fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=0)

        # Prédictions
        y_pred = self.model.predict(X_test)
        y_pred = self.scaler_y.inverse_transform(y_pred)
        y_test = self.scaler_y.inverse_transform(y_test)

        # Génération des signaux basés sur la différence de prix prédite
        signals = pd.Series(0, index=original_index)
        price_diff = np.diff(y_pred.flatten())
        test_signals = np.zeros(len(y_pred))
        test_signals[1:] = (price_diff > 0).astype(int) - (price_diff < 0).astype(int)
        signals.iloc[-test_size:] = test_signals

        self.signals = signals
        return signals

