# TimeSessionProject

### Телеграм-бот для анализа и прогнозирования акций на основе временных рядов

1 для запуска бота нужно создать токен "BOT_TOKEN" для телеграмм и добавить его в env

2 Запустить скрипт main.py

Для каждого запроса происходит выкачивание данных из yfinance и расчет на 3 моделях (ARIMA, Ridge, SimpleRNN) потенциальной прибыли выбирается по метрикам лучшая
и возврощается ответ. 

Пример метрик 

<img width="286" height="99" alt="image" src="https://github.com/user-attachments/assets/667a353b-e1f4-4928-9318-c1d4f3f151da" />

Пример работы бота

![photo_2025-11-10 17 28 57](https://github.com/user-attachments/assets/ae6c0317-f0dc-426f-b5ae-2eb476108a5e)


