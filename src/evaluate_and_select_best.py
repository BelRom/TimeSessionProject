
def evaluate_and_select_best(rmse_ridge, mape_ridge, rmse_arima, mape_arima, rmse_rnn, mape_rnn):
    """
    Выводит метрики моделей и выбирает лучшую по RMSE.
    """
    print(f"Ridge: RMSE={rmse_ridge:.3f}, MAPE={mape_ridge:.2f}%")
    print(f"ARIMA: RMSE={rmse_arima:.3f}, MAPE={mape_arima:.2f}%")
    print(f"RNN:    RMSE={rmse_rnn:.3f}, MAPE={mape_rnn:.2f}%")

    rmse_values = {"Ridge": rmse_ridge, "ARIMA": rmse_arima, "RNN": rmse_rnn}
    best_model = min(rmse_values, key=rmse_values.get)
    print(f"Best model: {best_model}")
    return best_model
