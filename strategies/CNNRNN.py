import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from strategies.strategy import Strategy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, LSTM, Dropout, Dense

class CNNRNNStrategy(Strategy):
    """CNN + RNN strategy for multi-step forecasting of prices or returns"""

    def __init__(self, window_size=20, n_steps_ahead=5, epochs=50, batch_size=32, target_type="price"):
        self.window_size = window_size
        self.n_steps_ahead = n_steps_ahead
        self.epochs = epochs
        self.batch_size = batch_size
        self.target_type = target_type  # 'price' or 'return'
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.signals = None

    def compute_indicators(self, df):
        df = df.copy()
        if self.target_type == "price":
            for i in range(self.n_steps_ahead):
                df[f'Target_{i+1}'] = df['Close'].shift(-i-1)
        elif self.target_type == "return":
            for i in range(1, self.n_steps_ahead + 1):
                df[f'Target_{i}'] = df['Close'].shift(-i) / df['Close'] - 1
        df.dropna(inplace=True)
        return df

    def generate_signals(self, df):
        original_index = df.index
        df = self.compute_indicators(df)
        features = ['High', 'Low', 'Close']

        # Normalisation
        X = self.scaler_X.fit_transform(df[features])
        y_cols = [f'Target_{i+1}' for i in range(self.n_steps_ahead)]
        y = df[y_cols].values
        y = self.scaler_y.fit_transform(y)

        # Création des séquences
        X_seq, y_seq = [], []
        for i in range(len(X) - self.window_size - self.n_steps_ahead + 1):
            X_seq.append(X[i:i + self.window_size])
            y_seq.append(y[i + self.window_size - 1])

        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)

        # Split
        test_size = int(len(X_seq) * 0.2)
        X_train, X_test = X_seq[:-test_size], X_seq[-test_size:]
        y_train, y_test = y_seq[:-test_size], y_seq[-test_size:]

        # 🧠 Architecture CNN + LSTM
        self.model = Sequential([
            Input(shape=(self.window_size, X_seq.shape[2])),
            Conv1D(filters=64, kernel_size=3, activation='relu'),
            BatchNormalization(),
            Conv1D(filters=32, kernel_size=3, activation='relu'),
            BatchNormalization(),
            LSTM(64, return_sequences=False),
            Dropout(0.2),
            Dense(self.n_steps_ahead)
        ])

        self.model.compile(optimizer='adam', loss='mse')
        self.model.fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size,
                       validation_split=0.2, verbose=0,
                       callbacks=[tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)])

        # Prédictions
        y_pred = self.model.predict(X_test)
        y_pred_inverse = self.scaler_y.inverse_transform(y_pred)
        y_test_inverse = self.scaler_y.inverse_transform(y_test)

        self.y_pred = y_pred_inverse
        self.y_test = y_test_inverse

        # Génération des signaux
        future_mean_pred = y_pred_inverse.mean(axis=1)
        reference = y_test_inverse[:, 0] if self.target_type == "price" else np.zeros_like(future_mean_pred)

        min_len = min(len(future_mean_pred), len(reference))
        future_mean_pred = future_mean_pred[-min_len:]
        reference = reference[-min_len:]

        test_signals = np.zeros(min_len)
        test_signals[future_mean_pred > reference] = 1
        test_signals[future_mean_pred < reference] = -1

        signals = pd.Series(0, index=original_index)
        test_start_idx = len(original_index) - len(test_signals)
        signals.iloc[test_start_idx:] = test_signals

        self.signals = signals
        return signals
