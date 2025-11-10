import yfinance as yf

# Функция для загрузки данных
def fetch_stock_data(ticker: str = "AAPL", period: str = "2y"):
    """
    Загружает данные о цене закрытия акций с Yahoo Finance.
    """

    data = yf.download(ticker, period=period, auto_adjust=True)
    if data.empty:
        raise ValueError(f"Не удалось загрузить данные по тикеру {ticker}.")

    data = data[["Close"]]
    data.dropna(inplace=True)
    return data
