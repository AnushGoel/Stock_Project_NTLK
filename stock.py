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
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
import nltk

nltk.download('vader_lexicon', quiet=True)

# ---------------------------
# Helper Functions
# ---------------------------
def fetch_news(ticker, limit=15):
    stock = yf.Ticker(ticker)
    return stock.news[:limit]

def sentiment_analysis(news):
    analyzer = SentimentIntensityAnalyzer()
    sentiments = [
        analyzer.polarity_scores(item['title'])['compound']
        for item in news if 'title' in item
    ]
    return np.mean(sentiments) if sentiments else 0.0

def summarize_text(text, sentences=2):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = TextRankSummarizer()
    summary = summarizer(parser.document, sentences)
    return " ".join([str(s) for s in summary])

def get_news_summaries(news_items):
    summaries = []
    for item in news_items:
        title = item.get('title', 'No Title')
        content = item.get('summary', title)
        try:
            summary = summarize_text(content)
        except:
            summary = content
        summaries.append({"Title": title, "Summary": summary})
    return pd.DataFrame(summaries)

def forecast_prophet(data, days):
    df = data.reset_index().rename(columns={"Date": "ds", "Close": "y"})
    df = df[['ds', 'y']].dropna()

    # Explicitly ensure numeric values
    df['y'] = pd.to_numeric(df['y'], errors='coerce')
    df.dropna(inplace=True)

    model = Prophet(daily_seasonality=True)
    model.fit(df)

    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)

    return forecast[['ds', 'yhat']].tail(days)['yhat'].values


def forecast_arima(series, days):
    model = ARIMA(series, order=(5,1,0))
    model_fit = model.fit()
    forecast = model_fit.forecast(days)
    return forecast

def forecast_lstm(data, days, seq_len=60):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data.values.reshape(-1, 1))
    X, y = [], []
    for i in range(seq_len, len(scaled)):
        X.append(scaled[i-seq_len:i, 0])
        y.append(scaled[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1],1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=10, batch_size=32, verbose=0)

    input_seq = scaled[-seq_len:].reshape(1, seq_len, 1)
    preds = []
    for _ in range(days):
        pred = model.predict(input_seq, verbose=0)
        preds.append(pred[0,0])
        input_seq = np.append(input_seq[:,1:,:],[[pred]], axis=1)
    return scaler.inverse_transform(np.array(preds).reshape(-1,1)).flatten()

# ---------------------------
# Streamlit UI Setup
# ---------------------------
st.set_page_config(page_title="Enhanced StockGPT 🚀", layout="wide")
st.title("📈 Enhanced StockGPT with Sentiment & News Summaries")

ticker = st.sidebar.text_input("Enter Stock Ticker:", "AAPL").upper()
start_date = st.sidebar.date_input("Start Date:", datetime.date.today() - datetime.timedelta(365))
end_date = datetime.date.today()
forecast_days = st.sidebar.slider("Forecast Days:", 7, 60, 14)

tabs = st.tabs(["Dashboard", "Charts", "Forecast", "News Summaries"])

data = yf.download(ticker, start=start_date, end=end_date)

news_items = fetch_news(ticker)
sentiment_score = sentiment_analysis(news_items)
sentiment_factor = 1 + (sentiment_score * 0.05)

with tabs[0]:
    st.header(f"{ticker} Overview")
    st.metric("News Sentiment Score", f"{sentiment_score:.2f}")
    recommendation = "🟢 Buy" if sentiment_score > 0 else "🔴 Hold/Sell"
    st.info(f"Investment Recommendation: {recommendation}")

with tabs[1]:
    st.header("📊 Historical Prices")
    
    chart_data = data.reset_index()[['Date', 'Close']].dropna()
    chart_data['Date'] = pd.to_datetime(chart_data['Date'])

    chart = alt.Chart(chart_data).mark_line().encode(
        x=alt.X('Date:T', title='Date'),
        y=alt.Y('Close:Q', title='Closing Price (USD)', axis=alt.Axis(format="$,.2f")),
        tooltip=['Date:T', alt.Tooltip('Close:Q', format="$,.2f")]
    ).properties(height=400, width=700, title=f"{ticker} Closing Prices Over Time")

    st.altair_chart(chart, use_container_width=True)


with tabs[2]:
    st.header("📉 Stock Price Forecast")

    prophet_pred = forecast_prophet(data, forecast_days)
    arima_pred = forecast_arima(data['Close'], forecast_days)
    lstm_pred = forecast_lstm(data['Close'], forecast_days)

    recent_actual = data['Close'][-forecast_days:].values
    errors = {
        "Prophet": mean_absolute_error(recent_actual, prophet_pred),
        "ARIMA": mean_absolute_error(recent_actual, arima_pred),
        "LSTM": mean_absolute_error(recent_actual, lstm_pred)
    }

    best_model = min(errors, key=errors.get)
    best_forecast = {"Prophet":prophet_pred,"ARIMA":arima_pred,"LSTM":lstm_pred}[best_model]
    adjusted_forecast = best_forecast * sentiment_factor

    forecast_dates = pd.date_range(end=end_date, periods=forecast_days)
    forecast_df = pd.DataFrame({
        "Date": forecast_dates,
        "Forecasted Price": adjusted_forecast.round(2)
    })

    st.success(f"Best Model: **{best_model}** (Adjusted for sentiment)")
    st.dataframe(forecast_df.style.format({"Forecasted Price": "${:,.2f}"}))

with tabs[3]:
    st.header("📰 Top News Summaries")
    news_df = get_news_summaries(news_items)
    for idx, row in news_df.iterrows():
        st.subheader(f"{idx+1}. {row['Title']}")
        st.write(row['Summary'])
