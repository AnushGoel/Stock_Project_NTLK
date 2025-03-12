import pandas as pd
import numpy as np

def calculate_RSI(data, window=14):
    """
    Calculate the Relative Strength Index (RSI) for the given data.
    """
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_MACD(data, short_window=12, long_window=26, signal_window=9):
    """
    Calculate the MACD and Signal line.
    Returns macd, signal, and histogram.
    """
    ema_short = data['Close'].ewm(span=short_window, adjust=False).mean()
    ema_long = data['Close'].ewm(span=long_window, adjust=False).mean()
    macd = ema_short - ema_long
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    histogram = macd - signal
    return macd, signal, histogram

def calculate_technical_indicators(data):
    """
    Calculates additional technical indicators and returns them as new columns.
    """
    data = data.copy()
    data['RSI'] = calculate_RSI(data)
    macd, signal, hist = calculate_MACD(data)
    data['MACD'] = macd
    data['MACD_Signal'] = signal
    data['MACD_Hist'] = hist
    return data
