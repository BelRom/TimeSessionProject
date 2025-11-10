import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

def forecast_future_prices(data, best_model_name, lags=5, future_period=30, ticker="AAPL"):
    """
    Обучает лучшую модель на всех доступных данных и строит прогноз на будущее.

    Возвращает:
        forecast (pd.Series): предсказанные значения
        buy_day (datetime or None), sell_day (datetime or None): рекомендации
    """
    full_series = data.xs('Close', axis=1, level='Price')[ticker].astype(float)
    full_series.name = "Close"
    last_date = full_series.index[-1]

    if best_model_name == "Ridge":
        df_full = full_series.to_frame()
        for i in range(1, lags + 1):
            df_full[f"lag_{i}"] = df_full["Close"].shift(i)
        df_full.dropna(inplace=True)

        feature_cols = [f"lag_{i}" for i in range(1, lags + 1)]
        X_full = df_full[feature_cols]
        y_full = df_full["Close"]

        model = Ridge().fit(X_full, y_full)
        future_preds = []
        window = list(full_series.values[-lags:])

        for _ in range(future_period):
            X_input = pd.DataFrame([window[-lags:]], columns=feature_cols)
            y_next = model.predict(X_input)[0]
            future_preds.append(y_next)
            window.append(y_next)

    elif best_model_name == "ARIMA":
        model = ARIMA(full_series, order=(5, 1, 0)).fit()
        future_preds = model.forecast(steps=future_period).to_numpy(dtype=float)

    elif best_model_name == "RNN":
        # Подготовка лагов
        df_full = full_series.to_frame(name="Close").copy()
        for i in range(1, lags + 1):
            df_full[f"lag_{i}"] = df_full["Close"].shift(i)
        df_full.dropna(inplace=True)

        feature_cols = [f"lag_{i}" for i in range(1, lags + 1)]
        X_full = df_full[feature_cols].values
        y_full = df_full["Close"].values.reshape(-1, 1)

        feature_scaler = MinMaxScaler()
        target_scaler = MinMaxScaler()

        X_scaled = feature_scaler.fit_transform(X_full).reshape(-1, lags, 1)
        y_scaled = target_scaler.fit_transform(y_full)

        model = Sequential()
        model.add(SimpleRNN(64, activation='tanh', input_shape=(lags, 1)))
        model.add(Dropout(0.3))
        model.add(Dense(1))
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        model.fit(X_scaled, y_scaled, epochs=100, batch_size=16, verbose=0)

        # Прогноз
        future_preds_scaled = []
        last_window = feature_scaler.transform(X_full[-1:].reshape(1, -1)).reshape(1, lags, 1)

        for _ in range(future_period):
            y_next_scaled = model.predict(last_window, verbose=0)[0, 0]
            future_preds_scaled.append(y_next_scaled)
            last_window = np.append(last_window[:, 1:, :], [[[y_next_scaled]]], axis=1)

        future_preds = target_scaler.inverse_transform(np.array(future_preds_scaled).reshape(-1, 1)).flatten()

    future_index = pd.bdate_range(last_date + pd.offsets.BDay(), periods=future_period)
    forecast = pd.Series(future_preds, index=future_index)

    # Поиск сигналов покупки и продажи
    vals = forecast.values
    buy_day = sell_day = None

    for i in range(1, len(vals) - 1):
        if vals[i] < vals[i - 1] and vals[i] < vals[i + 1]:
            buy_day = forecast.index[i]
            break

    if buy_day:
        for j in range(i + 1, len(vals) - 1):
            if vals[j] > vals[j - 1] and vals[j] > vals[j + 1]:
                sell_day = forecast.index[j]
                break

    return forecast, buy_day, sell_day
