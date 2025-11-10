import warnings
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings("ignore")

def rnn_forecast(X_train, y_train, X_test, y_test, lags=12):
    """
    Обучает RNN-модель и делает многошаговый прогноз на тестовой выборке.
    """
    # Масштабирование и reshape
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    X_scaled = feature_scaler.fit_transform(X_train)
    y_scaled = target_scaler.fit_transform(y_train.values.reshape(-1, 1))

    X_rnn = X_scaled.reshape((X_scaled.shape[0], lags, 1))

    # Валидация через TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=3)
    best_model = None
    best_val_loss = float("inf")

    for train_index, val_index in tscv.split(X_rnn):
        X_t, X_val = X_rnn[train_index], X_rnn[val_index]
        y_t, y_val = y_scaled[train_index], y_scaled[val_index]

        model = Sequential()
        model.add(SimpleRNN(64, activation='tanh', input_shape=(lags, 1)))
        model.add(Dropout(0.3))
        model.add(Dense(1))
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

        early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        model.fit(X_t, y_t, epochs=100, batch_size=16, validation_data=(X_val, y_val),
                  callbacks=[early_stop], verbose=0)

        val_loss = model.evaluate(X_val, y_val, verbose=0)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model

    # Прогноз на тесте
    preds_scaled = []
    lag_vector = feature_scaler.transform(X_test.iloc[0:1]).reshape(1, lags, 1)

    for _ in range(len(y_test)):
        y_pred_scaled = best_model.predict(lag_vector, verbose=0)[0, 0]
        preds_scaled.append(y_pred_scaled)
        lag_vector = np.append(lag_vector[:, 1:, :], [[[y_pred_scaled]]], axis=1)

    preds = target_scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mape = mean_absolute_percentage_error(y_test, preds) * 100

    return preds, rmse, mape
