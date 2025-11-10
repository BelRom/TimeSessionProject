import warnings
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")  # отключаем предупреждения для чистоты вывода

def arima_forecast(y_train, y_test, order=(5, 1, 0)):
    """
    Обучает модель ARIMA и делает прогноз на тестовый горизонт.
    """
    # Обучение модели
    model = ARIMA(y_train, order=order)
    result = model.fit()

    # Прогноз
    horizon = len(y_test)
    preds = result.forecast(steps=horizon)
    preds = np.array(preds, dtype=float)

    # Метрики
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mape = mean_absolute_percentage_error(y_test, preds) * 100

    return preds, rmse, mape
