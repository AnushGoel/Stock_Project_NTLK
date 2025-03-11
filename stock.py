import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime
import altair as alt
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from gensim.summarization import summarize
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

nltk.download('vader_lexicon')
vader = SentimentIntensityAnalyzer()

# ---------------------------
# Fetch News & Sentiment Analysis
# ---------------------------
def fetch_news(ticker):
    stock = yf.Ticker(ticker)
    return stock.news[:15]  # Get latest 15 news

def analyze_sentiment(news_items):
    analyzer = SentimentIntensityAnalyzer()
    sentiments = [analyzer.polarity_scores(item['title'])['compound'] for item in news_items]
    avg_sentiment = np.mean(sentiments) if sentiments else 0
    return avg_sentiment, sentiments

def summarize_news(news_items):
    summaries = []
    for item in news_items:
        content = item.get('summary', item['title'])
        try:
            summary = summarize(content, word_count=30)
            summaries.append({"title": item['title'], "summary": summary})
        except ValueError:
            summaries.append({"title": item['title'], "summary": item['title']})
    return pd.DataFrame(summaries)

# ---------------------------
# Streamlit UI Setup
# ---------------------------
st.set_page_config(page_title="StockGPT Dashboard 🚀", layout="wide")

st.sidebar.title("Settings")
ticker = st.sidebar.text_input("Ticker", "AAPL").upper()
start = st.sidebar.date_input("Start date", datetime.date.today() - datetime.timedelta(days=365))
end_date = datetime.date.today()
forecast_days = st.sidebar.slider("Forecast Days", 7, 60, 14)

tabs = st.tabs(["Dashboard", "Charts", "Forecast", "News Summary", "Chat Advisor"])

# Fetch data
data = yf.download(ticker, start=start, end=end_date)

# ---------------------------
# Dashboard
# ---------------------------
with tabs[0]:
    st.header(f"{ticker} - Stock Overview")
    comp_name = yf.Ticker(ticker).info.get('longName', ticker)
    st.write(f"**{comp_name}**")
    sentiment = analyze_sentiment(fetch_news(ticker))
    st.write(f"News Sentiment: {sentiment:.2f}")
    recommendation = "🟢 Buy" if sentiment > 0 else "🔴 Caution/Hold"
    st.write(f"**Investment Recommendation:** {recommendation}")

# ---------------------------
# Charts
# ---------------------------
with tabs[1]:
    st.header("📊 Stock Data Visualization")
    chart = alt.Chart(data.reset_index()).mark_line().encode(x='Date', y='Close')
    st.altair_chart(chart, use_container_width=True)

# ---------------------------
# Forecast Tab
# ---------------------------
with tabs[2]:
    st.header("📈 Stock Price Forecast")
    
    # Prophet Forecast
    df_prophet = data.reset_index().rename(columns={'Date': 'ds', 'Close': 'y'})
    model_prophet = Prophet(daily_seasonality=True)
    model_prophet.fit(df_prophet)
    future = model_prophet.make_future_dataframe(periods=forecast_days)
    forecast_prophet = model_prophet.predict(future).tail(forecast_days)['yhat'].values

    # Adjust forecasts based on sentiment
    sentiment_multiplier = 1 + (sentiment / 10)
    adjusted_forecast = forecast = forecast_prophet * (1 + sentiment * 0.05)

    forecast_df = pd.DataFrame({
        "Date": pd.date_range(end=end_date, periods=forecast_days),
        "Forecasted Price": adjusted_forecast.round(2)
    })

    # Visualize Forecast
    st.dataframe(forecast_df.style.format({"Forecasted Price": "${:,.2f}"}))
    chart = alt.Chart(forecast_df).mark_line(color='orange').encode(
        x='Date:T',
        y=alt.Y('Forecasted Price', axis=alt.Axis(format="$,.2f")),
        tooltip=['Date', 'Forecasted Price']
    ).properties(title=f"{ticker} Price Forecast")
    st.altair_chart(chart, use_container_width=True)

    st.write("Forecast prices are adjusted according to news sentiment.")

# ---------------------------
# News Summary
# ---------------------------
with tabs[3]:
    st.header("📰 Top 15 News Impacting Stock Prices")
    news_items = fetch_data(ticker, start_date, end_date)
    news_summary_df = summarize_news(news_items)
    for idx, row in news_summary.iterrows():
        st.subheader(f"🔹 {row['title']}")
        st.write(row['summary'])

# ---------------------------
# Chat Advisor
# ---------------------------
with tabs[4]:
    st.header("💬 Your Chat Advisor")
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    def process_chat(query):
        query = query.lower()
        if "price" in query:
            return f"Today's price for {ticker} is ${data['Close'][-1]:,.2f}"
        elif "sentiment" in query:
            return f"Current sentiment score is {sentiment:.2f} ({'Positive' if sentiment > 0 else 'Negative/Neutral'})."
        else:
            return "Please provide a specific query about stock analysis."

    user_query = st.text_input("Ask me about stock analysis:")
    if st.button("Send"):
        st.session_state["chat_history"].append(("You", user_query))
        response = process_chat(user_query)
        st.session_state["chat_history"].append(("Advisor", response))

    for speaker, message in st.session_state["chat_history"]:
        st.markdown(f"**{speaker}:** {message}")

