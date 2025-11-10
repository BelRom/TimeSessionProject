import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

def ridge_forecast(X_train, y_train, X_test, y_test, lags=12, alpha=1.0):
    """
    Обучает модель Ridge и делает многошаговый прогноз по лаговым признакам.
    """
    feature_cols = [f"lag_{i}" for i in range(1, lags + 1)]

    # Обучение модели
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)

    # Многошаговый прогноз
    horizon = len(y_test)
    preds = []
    lag_vector = X_test.iloc[0].to_numpy(dtype=float)

    for _ in range(horizon):
        X_input = pd.DataFrame(lag_vector.reshape(1, -1), columns=feature_cols)
        y_pred = model.predict(X_input)[0]
        preds.append(y_pred)
        lag_vector = np.roll(lag_vector, -1)
        lag_vector[-1] = y_pred

    preds = np.array(preds, dtype=float)

    # Оценка качества
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mape = mean_absolute_percentage_error(y_test, preds) * 100

    return preds, rmse, mape
