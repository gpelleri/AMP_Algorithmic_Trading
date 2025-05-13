import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Dropout
from tensorflow.keras.optimizers import Adam
from strategies.strategy import Strategy

class SimpleRNNStrategy(Strategy):
    """
    Simple RNN Strategy for single-period price prediction.
    """
    def __init__(self, lookback=30, n_splits=5, epochs=30):
        self.lookback = lookback  # number of past time steps to use
        self.n_splits = n_splits
        self.epochs = epochs
        self.model = None
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.signals = None
        self.predictions = None
        self.cv_scores = []

    def prepare_data(self, data):
        # Scale the close prices
        close_prices = data['Close'].values.reshape(-1, 1)
        scaled_prices = self.scaler_X.fit_transform(close_prices)
        
        X = []
        y = []
        
        # Create sequences of lookback days
        for i in range(len(scaled_prices) - self.lookback):
            X.append(scaled_prices[i:(i + self.lookback)])
            y.append(scaled_prices[i + self.lookback])
            
        X = np.array(X)
        y = np.array(y)
        
        return X, y

    def build_model(self):
        model = Sequential([
            SimpleRNN(50, activation='tanh', input_shape=(self.lookback, 1), return_sequences=False),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model

    def cross_validate_and_train(self, X, y):
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        cv_scores = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model = self.build_model()
            model.fit(X_train, y_train, epochs=self.epochs, batch_size=32, 
                     validation_data=(X_val, y_val), verbose=0)
            
            val_score = model.evaluate(X_val, y_val, verbose=0)
            cv_scores.append(val_score)
        
        self.cv_scores = cv_scores
        # Train final model on all data
        self.model = self.build_model()
        self.model.fit(X, y, epochs=self.epochs, batch_size=32, verbose=0)

    def generate_prediction(self, X_new):
        return self.model.predict(X_new, verbose=0)

    def generate_signals(self, data):
        X, y = self.prepare_data(data)
        if self.model is None:
            self.cross_validate_and_train(X, y)
        
        # Generate predictions
        predictions = self.generate_prediction(X)
        predictions = self.scaler_X.inverse_transform(predictions)
        
        # Create signals DataFrame
        signals = pd.Series(0, index=data.index)
        
        # Offset predictions to align with the right dates
        prediction_index = data.index[self.lookback:]
        actual_prices = data['Close'].loc[prediction_index]
        pred_prices = pd.Series(predictions.flatten(), index=prediction_index)
        
        # Generate trading signals
        signals[prediction_index] = np.where(pred_prices > actual_prices, 1, -1)
        
        self.signals = signals
        return signals

    def plot_signals(self, data):
        # Implementation for plotting signals
        pass
