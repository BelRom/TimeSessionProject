from telegram import Update
from telegram.ext import (
    Updater,
    CommandHandler,
    MessageHandler,
    Filters,
    CallbackContext,
    ConversationHandler,
)
from datetime import datetime
import logging
from dotenv import load_dotenv
import os

from src.evaluate_and_select_best import evaluate_and_select_best
from src.fetch_stock_data import fetch_stock_data
from src.model_arima_forecast import arima_forecast
from src.model_ridge_forecast import ridge_forecast
from src.model_rnn_forecast import rnn_forecast
from src.predict_forecast_future_prices import forecast_future_prices
from src.prepare_lag_features_test_split import prepare_lag_features

load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")
TICKER, AMOUNT = range(2)

# Логгер
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)

# Команда /start
def start(update: Update, context: CallbackContext) -> int:
    update.message.reply_text("Привет! Введите код акции (например, AAPL):")
    return TICKER

# Получение кода акции
def handle_ticker(update: Update, context: CallbackContext) -> int:
    ticker = update.message.text.strip().upper()
    context.user_data["ticker"] = ticker
    update.message.reply_text(f"Код акции {ticker} принят. Теперь введите сумму инвестиции:")
    return AMOUNT

# Получение суммы
def handle_amount(update: Update, context: CallbackContext) -> int:
    try:
        amount = float(update.message.text.replace(",", "."))
        ticker = context.user_data.get("ticker")
        user_id = update.message.from_user.id
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        update.message.reply_text(
            f"Вы указали: {ticker}, сумма: {amount:.2f}.\n"
        )

        result = calculate_possible_profit(ticker = ticker, amount = amount)
        # Лог в CSV
        with open("requests_log.csv", "a", encoding="utf-8") as f:
            f.write(f"{ticker},{amount},{timestamp},{user_id},{result}\n")
        update.message.reply_text(result)
        return ConversationHandler.END
    except ValueError:
        update.message.reply_text("Пожалуйста, введите корректное число.")
        return AMOUNT

# Команда /cancel
def cancel(update: Update, context: CallbackContext) -> int:
    update.message.reply_text("Диалог отменён. Чтобы начать заново, введите /start.")
    return ConversationHandler.END

def calculate_possible_profit(ticker: str, amount: float = 1000.0):
    data = fetch_stock_data(ticker)
    lags = 5
    X_train, y_train, X_test, y_test, feature_cols = prepare_lag_features(data, ticker, lags)
    preds_arima, rmse_arima, mape_arima = arima_forecast(y_train, y_test)
    preds_ridge, rmse_ridge, mape_ridge = ridge_forecast(X_train, y_train, X_test, y_test, lags)
    preds_rnn, rmse_rnn, mape_rnn = rnn_forecast(X_train, y_train, X_test, y_test, lags)
    best_model_name = evaluate_and_select_best(rmse_ridge, mape_ridge, rmse_arima, mape_arima, rmse_rnn, mape_rnn)
    forecast, buy_day, sell_day = forecast_future_prices(data, best_model_name, lags=lags, future_period=30, ticker=ticker)
    profit = ((forecast[sell_day] - forecast[buy_day]) / forecast[buy_day]) * amount

    buy_price = forecast[buy_day]
    sell_price = forecast[sell_day]
    buy_str = buy_day.strftime("%d.%m")
    sell_str = sell_day.strftime("%d.%m")

    return (f"Покупка: {buy_str} по цене {buy_price:.2f}\n"
            f"Продажа: {sell_str} по цене {sell_price:.2f}\n"
            f"Прибыль: {profit:.2f}\n")

def run_bot():
    updater = Updater(BOT_TOKEN, use_context=True)
    dp = updater.dispatcher

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            TICKER: [MessageHandler(Filters.text & ~Filters.command, handle_ticker)],
            AMOUNT: [MessageHandler(Filters.text & ~Filters.command, handle_amount)],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )

    dp.add_handler(conv_handler)

    print("Бот запущен.")
    updater.start_polling()
    updater.idle()

