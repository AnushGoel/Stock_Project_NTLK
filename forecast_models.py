# forecast_models.py
import pandas as pd
import numpy as np
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def forecast_prophet(data, forecast_days, tuned_params=None):
    """
    Forecast future stock prices using Prophet.
    Returns a DataFrame with columns: forecast, lower, upper.
    """
    try:
        # Flatten data if needed and ensure Date is a column.
        df = data.reset_index()[['Date', 'Close']].copy()
        df.rename(columns={"Close": "y"}, inplace=True)
        seasonality_mode = tuned_params.get("seasonality_mode", "additive") if tuned_params else "additive"
        model = Prophet(daily_seasonality=True, seasonality_mode=seasonality_mode)
        model.fit(df)
        future = model.make_future_dataframe(periods=forecast_days)
        forecast = model.predict(future)
        result = forecast[['yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_days)
        result = result.rename(columns={"yhat": "forecast", "yhat_lower": "lower", "yhat_upper": "upper"})
        logging.info("Prophet forecast for %d days computed", forecast_days)
        return result
    except Exception as e:
        logging.error("Error in forecast_prophet: %s", e)
        return pd.DataFrame({"forecast": np.zeros(forecast_days),
                             "lower": np.zeros(forecast_days),
                             "upper": np.zeros(forecast_days)})

def forecast_arima(series, forecast_days, tuned_params=None):
    """
    Forecast future stock prices using an ARIMA model.
    Returns a DataFrame with columns: forecast, lower, upper.
    """
    try:
        order = tuned_params.get("order", (5,1,0)) if tuned_params else (5,1,0)
        # Ensure series is a 1D pd.Series
        series = pd.Series(series)
        model = ARIMA(series, order=order)
        model_fit = model.fit()
        forecast_result = model_fit.get_forecast(steps=forecast_days)
        forecast_values = forecast_result.predicted_mean
        conf_int = forecast_result.conf_int()
        result = pd.DataFrame({
            "forecast": forecast_values,
            "lower": conf_int.iloc[:, 0],
            "upper": conf_int.iloc[:, 1]
        })
        logging.info("ARIMA forecast for %d days computed", forecast_days)
        return result
    except Exception as e:
        logging.error("Error in forecast_arima: %s", e)
        return pd.DataFrame({"forecast": np.zeros(forecast_days),
                             "lower": np.zeros(forecast_days),
                             "upper": np.zeros(forecast_days)})

def forecast_lstm(series, forecast_days, tuned_params=None):
    """
    Forecast future stock prices using an LSTM model.
    Returns a DataFrame with columns: forecast, lower, upper.
    Confidence interval is simulated as ±5% of forecast.
    """
    try:
        data_vals = series.values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data_vals)
        
        time_step = 60
        X, y = [], []
        for i in range(time_step, len(scaled_data)):
            X.append(scaled_data[i-time_step:i, 0])
            y.append(scaled_data[i, 0])
        X, y = np.array(X), np.array(y)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        epochs = tuned_params.get("epochs", 10) if tuned_params else 10
        batch_size = tuned_params.get("batch_size", 32) if tuned_params else 32

        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
        model.add(LSTM(50))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)
        
        temp_input = scaled_data[-time_step:].tolist()
        lst_output = []
        for i in range(forecast_days):
            x_input = np.array(temp_input[-time_step:])
            x_input = x_input.reshape(1, time_step, 1)
            yhat = model.predict(x_input, verbose=0)
            lst_output.append(yhat[0][0])
            temp_input.append([yhat[0][0]])
        forecast_values = scaler.inverse_transform(np.array(lst_output).reshape(-1, 1)).flatten()
        lower = forecast_values * 0.95
        upper = forecast_values * 1.05
        result = pd.DataFrame({
            "forecast": forecast_values,
            "lower": lower,
            "upper": upper
        })
        logging.info("LSTM forecast for %d days computed", forecast_days)
        return result
    except Exception as e:
        logging.error("Error in forecast_lstm: %s", e)
        return pd.DataFrame({"forecast": np.zeros(forecast_days),
                             "lower": np.zeros(forecast_days),
                             "upper": np.zeros(forecast_days)})
