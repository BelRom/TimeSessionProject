
def prepare_lag_features(data, ticker="AAPL", lags=5, horizon=30):
    """
    Формирует лаговые признаки и делит на train/test для указанного тикера.
    """
    # Извлекаем ряд Close
    close_series = data.xs('Close', axis=1, level='Price')[ticker]
    close_series = close_series.astype(float)
    close_series.name = "Close"

    # Формируем лаговые признаки
    df_lags = close_series.to_frame()
    for i in range(1, lags + 1):
        df_lags[f"lag_{i}"] = df_lags["Close"].shift(i)
    df_lags = df_lags.dropna()

    # Разделяем на train/test
    train_df = df_lags.iloc[:-horizon]
    test_df  = df_lags.iloc[-horizon:]

    feature_cols = [f"lag_{i}" for i in range(1, lags + 1)]

    X_train = train_df[feature_cols]
    y_train = train_df["Close"]
    X_test  = test_df[feature_cols]
    y_test  = test_df["Close"]

    return X_train, y_train, X_test, y_test, feature_cols
