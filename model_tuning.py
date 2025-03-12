def tune_prophet(data):
    """
    Simulate fast hyper-parameter tuning for Prophet.
    Returns tuned parameters as a dictionary.
    """
    # In a real scenario, you might search over different seasonality modes, changepoint prior scales, etc.
    return {"seasonality_mode": "multiplicative"}

def tune_arima(series):
    """
    Simulate fast hyper-parameter tuning for ARIMA.
    Returns tuned order as a dictionary.
    """
    # In a real scenario, you might run grid search over orders.
    return {"order": (5, 1, 0)}

def tune_lstm(series):
    """
    Simulate fast hyper-parameter tuning for LSTM.
    Returns tuned parameters.
    """
    # In a real scenario, you might search over epochs, batch sizes, etc.
    return {"epochs": 10, "batch_size": 32}
