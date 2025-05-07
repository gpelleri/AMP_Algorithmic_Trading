import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from strategies.strategy import Strategy
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.volatility import BollingerBands

class LSTMMultiPeriodStrategy(Strategy):
    """LSTM pour prédictions multi-périodes avec choix entre prix et rendements"""

    def __init__(self, window_size=20, forecast_horizon=5, predict_returns=False, epochs=50, batch_size=32):
        self.window_size = window_size
        self.forecast_horizon = forecast_horizon
        self.predict_returns = predict_returns
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.signals = None

    def compute_indicators(self, df):
        df = df.copy()
        
        # Créer un DataFrame pour les cibles futures
        future_targets = pd.DataFrame(index=df.index)
        
        # Calculer les cibles futures
        if self.predict_returns:
            returns = df['Close'].pct_change()
            for i in range(self.forecast_horizon):
                future_targets[f'target_{i}'] = returns.shift(-i)
        else:
            close_prices = df['Close']
            for i in range(self.forecast_horizon):
                future_targets[f'target_{i}'] = close_prices.shift(-i)

        # Fusionner avec le DataFrame original
        df = pd.concat([df, future_targets], axis=1)

        # Indicateurs techniques
        df['RSI'] = RSIIndicator(df['Close']).rsi()
        df['MACD'] = MACD(df['Close']).macd_diff()
        df['BB_width'] = BollingerBands(df['Close']).bollinger_wband()
        df['SMA_20'] = SMAIndicator(df['Close'], window=20).sma_indicator()
        df['EMA_20'] = EMAIndicator(df['Close'], window=20).ema_indicator()
        df['HL_PCT'] = (df['High'] - df['Low']) / df['Close']
        df['Return'] = df['Close'].pct_change()

        df.dropna(inplace=True)
        
        # Retourner le DataFrame et les noms des colonnes cibles
        target_columns = [f'target_{i}' for i in range(self.forecast_horizon)]
        return df, target_columns

    def create_sequences(self, X_data, y_data):
        """
        Crée des séquences pour l'entraînement LSTM
        X_data : features (N, num_features)
        y_data : targets (N, forecast_horizon)
        """
        X, y = [], []
        for i in range(len(X_data) - self.window_size - self.forecast_horizon + 1):
            # Séquence d'entrée
            X.append(X_data[i:(i + self.window_size)])
            # Cible : next forecast_horizon values
            y.append(y_data[i + self.window_size])
        return np.array(X), np.array(y)

    def generate_signals(self, df):
        original_index = df.index
        df, target_columns = self.compute_indicators(df)

        features = ['RSI', 'MACD', 'BB_width', 'SMA_20', 'EMA_20', 'Return', 'HL_PCT']
        X = self.scaler_X.fit_transform(df[features])
        
        # Préparation des cibles
        if self.predict_returns:
            y = df[target_columns].values
        else:
            y = self.scaler_y.fit_transform(df[target_columns].values)

        # Création des séquences
        X_seq, y_seq = self.create_sequences(X, y)

        test_size = int(len(X_seq) * 0.2)
        X_train = X_seq[:-test_size]
        X_test = X_seq[-test_size:]
        y_train = y_seq[:-test_size]
        y_test = y_seq[-test_size:]

        # Architecture LSTM
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(100, return_sequences=True, input_shape=(self.window_size, len(features))),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(50),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(self.forecast_horizon)
        ])
        self.model.compile(optimizer='adam', loss='mse')
        
        # Entraînement
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True
        )
        self.model.fit(X_train, y_train, 
                      epochs=self.epochs, 
                      batch_size=self.batch_size,
                      validation_split=0.2,
                      callbacks=[early_stopping],
                      verbose=0)

        # Prédictions
        y_pred = self.model.predict(X_test)
        
        # Inverse transform si nécessaire
        if not self.predict_returns:
            y_pred = self.scaler_y.inverse_transform(y_pred)
            y_test = self.scaler_y.inverse_transform(y_test)
        
        # Sauvegarder pour analyse
        self.y_pred = y_pred
        self.y_test = y_test

        # Générer les signaux
        if self.predict_returns:
            pred_trends = np.mean(y_pred > 0, axis=1)
        else:
            pred_trends = np.mean(y_pred[:, 1:] > y_pred[:, :-1], axis=1)
        
        test_signals = (pred_trends > 0.5).astype(int) - (pred_trends < 0.5).astype(int)
        
        signals = pd.Series(0, index=original_index)
        test_start_idx = len(original_index) - len(test_signals)
        signals.iloc[test_start_idx:] = test_signals

        self.signals = signals
        return signals