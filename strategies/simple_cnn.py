import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from strategies.strategy import Strategy

class SimpleCNNStrategy(Strategy):
    """
    Simplified CNN Strategy for price prediction with basic technical indicators.
    """
    def __init__(self, window_size=20, n_splits=3, epochs=30, batch_size=32, patience=5,
             forecast_horizon=1, predict_returns=False):
        self.window_size = window_size
        self.n_splits = n_splits
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.forecast_horizon = forecast_horizon
        self.predict_returns = predict_returns

        self.model = None
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.signals = None
        self.predictions = None
        self.cv_scores = []


    def calculate_indicators(self, data):
        """Calculate a minimal set of technical indicators."""
        df = data.copy()
        
        # RSI avec fenêtre plus courte
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=7).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=7).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Moyennes mobiles courtes
        df['SMA_5'] = df['Close'].rolling(window=5).mean()
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        
        # MACD simplifié
        df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
        
        return df.fillna(method='bfill')

    def prepare_data(self, data):
        df = self.calculate_indicators(data)
        features = ['Close', 'RSI', 'SMA_5', 'SMA_10', 'MACD']
        X_data = df[features].values
        y_data = df['Close'].values

        # Calcul du target
        if self.predict_returns:
            y_data = pd.Series(np.log(y_data)).diff(self.forecast_horizon).shift(-self.forecast_horizon).values
        else:
            y_data = np.roll(y_data, -self.forecast_horizon)

        # Nettoyage
        valid_mask = ~np.isnan(X_data).any(axis=1) & ~np.isnan(y_data)
        X_data = X_data[valid_mask]
        y_data = y_data[valid_mask]

        # Normalisation
        X_scaled = self.scaler_X.fit_transform(X_data)
        y_scaled = self.scaler_y.fit_transform(y_data.reshape(-1, 1))

        # Fenêtres glissantes
        X_windows, y_out = [], []
        for i in range(len(X_scaled) - self.window_size + 1):
            if i + self.window_size - 1 + self.forecast_horizon >= len(y_scaled):
                break
            X_windows.append(X_scaled[i:i + self.window_size])
            y_out.append(y_scaled[i + self.window_size - 1])

        return np.array(X_windows), np.array(y_out), valid_mask

    def build_model(self, input_shape):
        """Build a simplified CNN model."""
        model = Sequential([
            # Une seule couche convolutive
            Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape),
            MaxPooling1D(pool_size=2),
            Dropout(0.2),
            
            # Flatten et deux couches denses
            Flatten(),
            Dense(20, activation='relu'),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model

    def cross_validate_and_train(self, X_windows, y):
        """Perform time series cross-validation with early stopping."""
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        cv_scores = []
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=self.patience,
            restore_best_weights=True
        )
        
        for train_idx, val_idx in tscv.split(X_windows):
            X_train, X_val = X_windows[train_idx], X_windows[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model = self.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
            model.fit(
                X_train, y_train,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_data=(X_val, y_val),
                callbacks=[early_stopping],
                verbose=0
            )
            
            val_score = model.evaluate(X_val, y_val, verbose=0)
            cv_scores.append(val_score)
        
        self.cv_scores = cv_scores
        
        # Train final model
        self.model = self.build_model(input_shape=(X_windows.shape[1], X_windows.shape[2]))
        self.model.fit(
            X_windows, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[early_stopping],
            verbose=0
        )

    def generate_signals(self, data):
        X_windows, y, valid_mask = self.prepare_data(data)

        if self.model is None:
            self.cross_validate_and_train(X_windows, y)

        predictions_scaled = self.model.predict(X_windows, verbose=0)
        predictions = self.scaler_y.inverse_transform(predictions_scaled).flatten()

        self.predictions = predictions

        # Créer signaux (1 = buy, -1 = sell)
        signals = np.zeros(len(predictions))
        signals[1:] = np.where(predictions[1:] > predictions[:-1], 1, -1)

        # Index correct pour les signaux
        valid_data = data[valid_mask]
        signal_index_start = self.window_size - 1
        signal_index_end = signal_index_start + len(signals)
        signal_indices = valid_data.index[signal_index_start:signal_index_end]

        self.signals = pd.Series(signals, index=signal_indices)
        return self.signals

    def plot_signals(self, data): 
        """Plot predictions and trading signals."""
        if self.signals is None or self.predictions is None:
            print("No signals or predictions available. Run generate_signals() first.")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

        # Index des signaux
        index = self.signals.index

        if self.predict_returns:
            # Rendements log
            actual_returns = np.log(data['Close']).diff().loc[index]
            ax1.plot(index, actual_returns, label='Real returns', color='blue')
            ax1.plot(index, self.predictions, label='Predicted returns', color='red', linestyle='--')
            ax1.set_title('Real vs Predicted Returns')
        else:
            actual_prices = data['Close'].loc[index]
            ax1.plot(index, actual_prices, label='Actual Prices', color='blue')
            ax1.plot(index, self.predictions, label='Predicted Prices', color='red', linestyle='--')
            ax1.set_title('Comparison of Actual and Predicted Prices')

        ax1.legend()
        ax1.grid(True)

        # Plot 2: Trading signals
        ax2.plot(index, self.signals, label='Signals', color='green')
        ax2.set_title('Trading Signals')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()

        # Display cross-validation scores
        print("\nCross-validation MSE scores:")
        for i, score in enumerate(self.cv_scores, 1):
            print(f"Fold {i}: {score:.6f}")
        print(f"MSE moyen: {np.mean(self.cv_scores):.6f} (±{np.std(self.cv_scores):.6f})")
