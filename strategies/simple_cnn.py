import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from strategies.strategy import Strategy

class SimpleCNNStrategy(Strategy):
    def __init__(self, window_size=20, forecast_horizon=1, n_splits=5,
                 epochs=30, batch_size=32, patience=5, predict_returns=False):
        self.window_size = window_size
        self.forecast_horizon = forecast_horizon
        self.n_splits = n_splits
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.predict_returns = predict_returns

        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.model = None
        self.trained = False
        self.cv_scores = []
        self.signals = None
        self.predictions = None

    def calculate_indicators(self, df):
        df['RSI'] = df['Close'].pct_change().rolling(window=14).mean()
        df['SMA_5'] = df['Close'].rolling(window=5).mean()
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
        return df

    def build_model(self, input_shape):
        model = Sequential([
            Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
            MaxPooling1D(pool_size=2),
            Dropout(0.2),
            Flatten(),
            Dense(50, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def cross_validate_and_train(self, X, y):
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        self.cv_scores = []

        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model = self.build_model(X.shape[1:])
            early_stopping = EarlyStopping(monitor='val_loss', patience=self.patience, restore_best_weights=True)

            model.fit(X_train, y_train, validation_data=(X_val, y_val),
                      epochs=self.epochs, batch_size=self.batch_size,
                      callbacks=[early_stopping], verbose=0)

            val_loss = model.evaluate(X_val, y_val, verbose=0)
            self.cv_scores.append(val_loss)

        # Train final model on all data
        self.model = self.build_model(X.shape[1:])
        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size,
                       callbacks=[EarlyStopping(monitor='loss', patience=self.patience, restore_best_weights=True)],
                       verbose=0)

    def fit(self, data):
        df = self.calculate_indicators(data)
        features = ['Close', 'RSI', 'SMA_5', 'SMA_10', 'MACD']
        X_data = df[features].values
        y_data = df['Close'].values

        if self.predict_returns:
            y_data = pd.Series(np.log(y_data)).diff(self.forecast_horizon).shift(-self.forecast_horizon).values
        else:
            y_data = np.roll(y_data, -self.forecast_horizon)

        valid_mask = ~np.isnan(X_data).any(axis=1) & ~np.isnan(y_data)
        X_data = X_data[valid_mask]
        y_data = y_data[valid_mask]

        self.scaler_X.fit(X_data)
        self.scaler_y.fit(y_data.reshape(-1, 1))

        X_scaled = self.scaler_X.transform(X_data)
        y_scaled = self.scaler_y.transform(y_data.reshape(-1, 1))

        X_windows, y_out = [], []
        for i in range(len(X_scaled) - self.window_size + 1):
            if i + self.window_size - 1 + self.forecast_horizon >= len(y_scaled):
                break
            X_windows.append(X_scaled[i:i + self.window_size])
            y_out.append(y_scaled[i + self.window_size - 1])

        X_windows = np.array(X_windows)
        y_out = np.array(y_out)

        self.cross_validate_and_train(X_windows, y_out)
        self.trained = True

    def generate_signals(self, data):
        df = self.calculate_indicators(data)
        features = ['Close', 'RSI', 'SMA_5', 'SMA_10', 'MACD']
        X_data = df[features].values
        y_data = df['Close'].values

        if self.predict_returns:
            y_data = pd.Series(np.log(y_data)).diff(self.forecast_horizon).shift(-self.forecast_horizon).values
        else:
            y_data = np.roll(y_data, -self.forecast_horizon)

        valid_mask = ~np.isnan(X_data).any(axis=1) & ~np.isnan(y_data)
        X_data = X_data[valid_mask]
        y_data = y_data[valid_mask]

        X_scaled = self.scaler_X.transform(X_data)
        y_scaled = self.scaler_y.transform(y_data.reshape(-1, 1))

        X_windows, y_out = [], []
        indices = []
        for i in range(len(X_scaled) - self.window_size + 1):
            target_idx = i + self.window_size - 1
            if target_idx + self.forecast_horizon >= len(y_scaled):
                break
            X_windows.append(X_scaled[i:i + self.window_size])
            y_out.append(y_scaled[target_idx])
            indices.append(df.index[valid_mask][target_idx])

        X_windows = np.array(X_windows)
        y_out = np.array(y_out)

        if not self.trained:
            self.cross_validate_and_train(X_windows, y_out)
            self.trained = True

        predictions = self.model.predict(X_windows).flatten()
        predictions = self.scaler_y.inverse_transform(predictions.reshape(-1, 1)).flatten()
        y_out_true = self.scaler_y.inverse_transform(y_out.reshape(-1, 1)).flatten()

        # Build partial signal frame on aligned prediction dates
        signals_partial = pd.DataFrame(index=indices)
        signals_partial['Predicted'] = predictions
        signals_partial['Actual'] = y_out_true
        signals_partial['Signal'] = np.where(predictions > y_out_true, 1, -1)

        # Reindex to full data length, fill missing with 0
        full_signals = pd.Series(0, index=df.index)
        full_signals.update(signals_partial['Signal'])

        self.signals = signals_partial  # keep original for analysis if needed
        self.predictions = signals_partial['Predicted']
        return full_signals

    def plot_signals(self, data):
        """Plot predictions and trading signals."""
        if self.signals is None or self.predictions is None:
            print("No signals or predictions available. Run generate_signals() first.")
            return

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

        # Index of signals/predictions
        index = self.signals.index

        if self.predict_returns:
            # Plot actual vs predicted returns
            actual_returns = np.log(data['Close']).diff().loc[index]
            ax1.plot(index, actual_returns, label='Real returns', color='blue')
            ax1.plot(index, self.predictions, label='Predicted returns', color='red', linestyle='--')
            ax1.set_title('Real vs Predicted Returns')
        else:
            # Plot actual vs predicted prices
            actual_prices = data['Close'].loc[index]
            ax1.plot(index, actual_prices, label='Actual Prices', color='blue')
            ax1.plot(index, self.predictions, label='Predicted Prices', color='red', linestyle='--')
            ax1.set_title('Comparison of Actual and Predicted Prices')

        ax1.legend()
        ax1.grid(True)

        # Plot ONLY signals (as step function for clarity)
        ax2.plot(index, self.signals, label='Signals', color='green', drawstyle='steps-post')
        ax2.set_title('Trading Signals')
        ax2.set_ylim(-0.1, 1.1)  # Optional: clean y-axis
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.show()

        # Display cross-validation scores
        print("\nCross-validation MSE scores:")
        for i, score in enumerate(self.cv_scores, 1):
            print(f"Fold {i}: {score:.6f}")
        print(f"Mean MSE: {np.mean(self.cv_scores):.6f} (±{np.std(self.cv_scores):.6f})")