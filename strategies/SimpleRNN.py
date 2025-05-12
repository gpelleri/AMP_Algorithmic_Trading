import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from strategies.strategy import Strategy

class SimpleRNNStrategy(Strategy):
    """A simple RNN strategy for time series prediction"""

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
        df['Target'] = df['Close'].shift(-1)  # prédiction du prix de clôture suivant
        df.dropna(inplace=True)
        return df

    def create_sequences(self, X_data, y_data):
        """Create sequences for RNN training"""
        X, y = [], []
        for i in range(len(X_data) - self.window_size):
            X.append(X_data[i:(i + self.window_size)])
            y.append(y_data[i + self.window_size])
        return np.array(X), np.array(y)

    def generate_signals(self, df):
        original_index = df.index
        df = self.compute_indicators(df)        # Use only available price data
        features = ['High', 'Low', 'Close']
        X = self.scaler_X.fit_transform(df[features])
        y = df['Target'].values.reshape(-1, 1)
        y = self.scaler_y.fit_transform(y)

        # Create sequences for RNN
        X_seq, y_seq = self.create_sequences(X, y)

        # Train/test split
        test_size = int(len(X_seq) * 0.2)
        X_train = X_seq[:-test_size]
        X_test = X_seq[-test_size:]
        y_train = y_seq[:-test_size]
        y_test = y_seq[-test_size:]

        # Simple RNN model architecture
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.SimpleRNN(100, input_shape=(self.window_size, len(features))),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(50, activation='relu'),
            tf.keras.layers.Dense(1)
        ])

        # Compile and train
        self.model.compile(optimizer='adam', loss='mse')
        
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )

        self.model.fit(X_train, y_train,
                      epochs=self.epochs,
                      batch_size=self.batch_size,
                      validation_split=0.2,
                      callbacks=[early_stopping],
                      verbose=0)        # Generate signals
        y_pred = self.model.predict(X_test)
        y_pred = self.scaler_y.inverse_transform(y_pred).flatten()  # Flatten to 1D array
        
        # Sauvegarde des prédictions et vérité terrain pour visualisation
        self.y_test = y_test
        self.y_pred = y_pred

        # Convert predictions to signals
        test_signals = np.zeros(len(y_pred))
        test_signals[1:] = (y_pred[1:] > y_pred[:-1]).astype(int) - \
                          (y_pred[1:] < y_pred[:-1]).astype(int)

        # Create final signal series
        signals = pd.Series(0, index=original_index)
        test_start_idx = len(original_index) - len(test_signals)
        signals.iloc[test_start_idx:] = test_signals

        self.signals = signals
        return signals
