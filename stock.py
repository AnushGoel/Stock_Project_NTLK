import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime
import altair as alt
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer

nltk.download('vader_lexicon', quiet=True)

# ---------------------------
# Sentiment Analysis Functions
# ---------------------------
def fetch_news(ticker, limit=15):
    stock = yf.Ticker(ticker)
    news_items = stock.news[:limit]
    return news_items

def sentiment_analysis(news):
    analyzer = SentimentIntensityAnalyzer()
    sentiments = [analyzer.polarity_scores(item['title'])['compound'] for item in news]
    return np.mean(sentiments)

# ---------------------------
# Text Summarization (Sumy)
# ---------------------------
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer

def summarize_text(text, sentences=2):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = TextRankSummarizer()
    summary = summarizer(parser.document, sentences)
    return ' '.join([str(sent) for sent in summary])

# ---------------------------
# Forecasting Functions
# ---------------------------
def forecast_prophet(data, days):
    df = data.reset_index().rename(columns={"Date": "ds", "Close": "y"})
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(days)
    forecast = model.predict(future)
    return forecast[['ds', 'yhat']].tail(days)['yhat'].values

def forecast_arima(series, days):
    model = ARIMA(series, order=(5,1,0))
    fit = model.fit()
    return fit.forecast(days)

def forecast_lstm(data, days, seq_len=60):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data.values.reshape(-1, 1))
    X, y = [], []
    for i in range(seq_len, len(scaled)):
        X.append(scaled[i-seq_len:i, 0])
        y.append(scaled[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    model = Sequential([LSTM(50, return_sequences=True, input_shape=(X.shape[1],1)),
                        LSTM(50),
                        Dense(1)])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=10, batch_size=32, verbose=0)

    preds = []
    input_seq = scaled[-seq_len:].reshape(1, seq_len, 1)
    for _ in range(days):
        pred = model.predict(input_seq, verbose=0)
        preds.append(pred[0,0])
        input_seq = np.roll(input_seq, -1)
        input_seq[0,-1,0] = pred
    return scaler.inverse_transform(np.array(preds).reshape(-1,1)).flatten()

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Enhanced StockGPT 🚀", layout="wide")
st.title("📈 Enhanced StockGPT with Sentiment & News Summaries")

ticker = st.sidebar.text_input("Enter Stock Ticker:", "AAPL").upper()
start_date = st.sidebar.date_input("Start Date:", datetime.date.today() - datetime.timedelta(365))
end_date = datetime.date.today()
forecast_days = st.sidebar.slider("Forecast Days:", 7, 60, 14)

tabs = st.tabs(["Dashboard", "Charts", "Forecast", "News Summaries"])

data = yf.download(ticker, start=start_date, end=end_date)

# Sentiment Adjustment
news_items = fetch_news(ticker)
sentiment_score = sentiment_analysis(news_items)
sentiment_factor = 1 + (sentiment_score * 0.05)

with tabs[0]:
    st.header(f"{ticker} Overview")
    st.metric("News Sentiment Score", f"{sentiment_score:.2f}", 
              delta_color="inverse")
    recommendation = "🟢 Positive Outlook" if sentiment_score > 0 else "🔴 Caution Advised"
    st.write(f"Investment Recommendation: {recommendation}")

with tabs[1]:
    st.header("📊 Historical Prices")
    chart = alt.Chart(data.reset_index()).mark_line().encode(
        x='Date', y='Close', tooltip=['Date', 'Close']).properties(height=400)
    st.altair_chart(chart, use_container_width=True)

with tabs[2]:
    st.header("📉 Stock Price Forecast")

    prophet_pred = forecast_prophet(data, forecast_days)
    arima_pred = forecast_arima(data['Close'], forecast_days)
    lstm_pred = forecast_lstm(data['Close'], forecast_days)

    actual_recent = data['Close'][-forecast_days:].values
    errors = {
        "Prophet": mean_absolute_error(actual_recent, prophet_pred),
        "ARIMA": mean_absolute_error(actual_recent, arima_pred),
        "LSTM": mean_absolute_error(actual_recent, lstm_pred)
    }
    best_model_name = min(errors, key=errors.get)
    best_forecast = {"Prophet": prophet_pred, "ARIMA": arima_pred, "LSTM": lstm_pred}[best_model_name]
    adjusted_forecast = best_forecast * sentiment_factor

    forecast_dates = pd.date_range(end=end_date, periods=forecast_days)
    forecast_df = pd.DataFrame({
        "Date": forecast_dates,
        "Forecasted Price": adjusted_forecast.round(2)
    })

    st.success(f"Best Model: **{best_model_name}** (Adjusted by Sentiment)")
    st.dataframe(forecast_df.style.format({"Forecasted Price": "${:,.2f}"}))

with tabs[3]:
    st.header("📰 News Summaries")
    summaries_list = []
    for item in news_items:
        title = item.get('title', '')
        summary = summarize_text(item.get('summary', title))
        summaries_list.append({"Title": title, "Summary": summary})

    news_df = pd.DataFrame(summaries_list)
    for idx, row in news_summary_df.iterrows():
        st.subheader(f"{idx+1}. {row['Title']}")
        st.write(row['Summary'])

