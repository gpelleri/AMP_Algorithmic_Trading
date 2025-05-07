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
from ta.volume import VolumeWeightedAveragePrice

class ANNStrategy(Strategy):
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
        df['Return'] = df['Close'].pct_change().shift(-1)
        df['RSI'] = RSIIndicator(df['Close']).rsi()
        df['MACD'] = MACD(df['Close']).macd_diff()
        df['BB_width'] = BollingerBands(df['Close']).bollinger_wband()
        df.dropna(inplace=True)
        return df

    def generate_signals(self, df):
        original_index = df.index  # Sauvegarder l'index original
        df = self.compute_indicators(df)
        features = ['RSI', 'MACD', 'BB_width']
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
        df['Return'] = df['Close'].pct_change().shift(-1)
        df['RSI'] = RSIIndicator(df['Close']).rsi()
        df['MACD'] = MACD(df['Close']).macd_diff()
        df['BB_width'] = BollingerBands(df['Close']).bollinger_wband()
        df.dropna(inplace=True)
        return df

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
            tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(self.window_size, X.shape[2])),
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
        df = df.copy()        # Indicateurs techniques supplémentaires
        df['RSI'] = RSIIndicator(df['Close']).rsi()
        df['MACD'] = MACD(df['Close']).macd_diff()
        df['BB_width'] = BollingerBands(df['Close']).bollinger_wband()
        df['SMA_20'] = SMAIndicator(df['Close'], window=20).sma_indicator()
        df['EMA_20'] = EMAIndicator(df['Close'], window=20).ema_indicator()
        # Indicateurs basés sur High/Low
        df['HL_PCT'] = (df['High'] - df['Low']) / df['Close']  # Volatilité High-Low
        df.dropna(inplace=True)
        return df

    def generate_signals(self, df):
        original_index = df.index
        df = self.compute_indicators(df)
        
        features = ['RSI', 'MACD', 'BB_width', 'SMA_20', 'EMA_20', 'HL_PCT']
        X = self.scaler_X.fit_transform(df[features])
        y = self.scaler_y.fit_transform(df[['Close']])

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

class LSTMMultiPeriodStrategy(Strategy):
    """LSTM pour prédictions multi-périodes"""
    def __init__(self, window_size=20, forecast_horizon=5, epochs=50, batch_size=32):
        self.window_size = window_size
        self.forecast_horizon = forecast_horizon
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

    def create_sequences(self, X_data, y_data):
        X, y = [], []
        for i in range(len(X_data) - self.window_size - self.forecast_horizon + 1):
            X.append(X_data[i:(i + self.window_size)])
            y.append(y_data[i + self.window_size:i + self.window_size + self.forecast_horizon])
        return np.array(X), np.array(y)

    def generate_signals(self, df):
        original_index = df.index
        df = self.compute_indicators(df)
        
        features = ['RSI', 'MACD', 'BB_width', 'SMA_20', 'EMA_20', 'Return']
        X = self.scaler_X.fit_transform(df[features])
        y = self.scaler_y.fit_transform(df[['Close']])

        X_seq, y_seq = self.create_sequences(X, y)
        
        test_size = int(len(X_seq) * 0.2)
        X_train = X_seq[:-test_size]
        X_test = X_seq[-test_size:]
        y_train = y_seq[:-test_size]
        y_test = y_seq[-test_size:]

        self.model = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(100, return_sequences=True, input_shape=(self.window_size, len(features))),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(50),
            tf.keras.layers.Dense(self.forecast_horizon)
        ])
        self.model.compile(optimizer='adam', loss='mse')
        self.model.fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=0)

        signals = pd.Series(0, index=original_index)
        y_pred = self.model.predict(X_test)
        
        # Générer des signaux basés sur la tendance moyenne prédite
        pred_trends = np.mean(y_pred > y_pred[:, 0:1], axis=1)
        test_signals = (pred_trends > 0.5).astype(int) - (pred_trends < 0.5).astype(int)
        
        test_start_idx = len(original_index) - len(test_signals)
        signals.iloc[test_start_idx:] = test_signals

        self.signals = signals
        return signals

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


    if __name__=="__main__":
        from utils.backtest import Backtest
        from utils.data_handler import DataHandler

        data_handler = DataHandler(ticker='AAPL', start_date='2020-01-01', end_date='2024-01-01')
        data = data_handler.fetch_data()

        my_strategy = ANNStrategy()
        backtest = Backtest(data, my_strategy)
        results = backtest.run()
        #my_strategy.plot_signals(data)
        print(results)