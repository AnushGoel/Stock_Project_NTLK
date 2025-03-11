# main.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import plotly.express as px
import plotly.graph_objects as go
import logging
import warnings

# Suppress warnings and configure logging
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Download NLTK VADER lexicon for sentiment analysis
nltk.download('vader_lexicon', quiet=True)

# =============================================================================
# Helper Functions
# =============================================================================

def fetch_news(ticker):
    """
    Fetch news articles for the given ticker.
    For demonstration, return dummy news data.
    In production, integrate with a real news API.
    """
    dummy_news = [
        {"title": f"{ticker} hits record high in US markets", 
         "content": f"The stock {ticker} has reached a record high in the US market due to strong earnings and positive investor sentiment. Analysts are optimistic about the growth prospects."},
        {"title": f"Concerns over {ticker}'s supply chain", 
         "content": f"Recent reports indicate potential disruptions in the supply chain for {ticker}, which could affect future performance and lead to a downturn in investor confidence."},
        {"title": f"{ticker} announces new product line", 
         "content": f"{ticker} is set to launch a new product line that is expected to boost sales and improve market share across the US and global markets."},
    ]
    logging.info("Fetched dummy news for ticker: %s", ticker)
    return dummy_news

def sentiment_analysis(news_items):
    """
    Compute the average sentiment score of provided news items using VADER.
    Returns a compound sentiment score.
    """
    analyzer = SentimentIntensityAnalyzer()
    scores = []
    for article in news_items:
        text = article.get("content", "")
        score = analyzer.polarity_scores(text)["compound"]
        scores.append(score)
    avg_score = np.mean(scores) if scores else 0.0
    logging.info("Calculated average sentiment score: %f", avg_score)
    return avg_score

def summarize_text(text, sentences_count=2):
    """
    Summarize text using TextRank algorithm from sumy.
    """
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = TextRankSummarizer()
    summary = summarizer(parser.document, sentences_count)
    summary_text = " ".join(str(sentence) for sentence in summary)
    logging.info("Summarized text: %s", summary_text)
    return summary_text

def get_news_summaries(news_items):
    """
    Create a DataFrame containing news titles and summaries.
    """
    summaries = []
    for article in news_items:
        title = article.get("title", "No Title")
        content = article.get("content", "")
        summary = summarize_text(content, sentences_count=2)
        summaries.append({"Title": title, "Summary": summary})
    df = pd.DataFrame(summaries)
    logging.info("Generated news summaries DataFrame with %d records", len(df))
    return df

def forecast_prophet(data, forecast_days):
    """
    Forecast future stock prices using the Prophet model.
    """
    df = data.reset_index()[['Date', 'Close']].rename(columns={"Date": "ds", "Close": "y"})
    model = Prophet(daily_seasonality=True)
    model.fit(df)
    future = model.make_future_dataframe(periods=forecast_days)
    forecast = model.predict(future)
    forecast_values = forecast['yhat'][-forecast_days:].values
    logging.info("Prophet forecast for %d days computed", forecast_days)
    return forecast_values

def forecast_arima(series, forecast_days):
    """
    Forecast future stock prices using an ARIMA model.
    """
    model = ARIMA(series, order=(5,1,0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=forecast_days)
    logging.info("ARIMA forecast for %d days computed", forecast_days)
    return forecast.values

def forecast_lstm(series, forecast_days):
    """
    Forecast future stock prices using an LSTM model.
    Uses the last 60 days as input to predict future prices.
    """
    data_vals = series.values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data_vals)
    
    time_step = 60
    X, y = [], []
    for i in range(time_step, len(scaled_data)):
        X.append(scaled_data[i-time_step:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X, y, epochs=10, batch_size=32, verbose=0)
    logging.info("LSTM model training complete")
    
    temp_input = scaled_data[-time_step:].tolist()
    lst_output = []
    for i in range(forecast_days):
        x_input = np.array(temp_input[-time_step:])
        x_input = x_input.reshape(1, time_step, 1)
        yhat = model.predict(x_input, verbose=0)
        lst_output.append(yhat[0][0])
        temp_input.append([yhat[0][0]])
    forecast_values = scaler.inverse_transform(np.array(lst_output).reshape(-1, 1)).flatten()
    logging.info("LSTM forecast for %d days computed", forecast_days)
    return forecast_values

def additional_interactive_features(data):
    """
    Create extra interactive elements for deeper data exploration.
    Returns a dictionary of data tables and Plotly figures.
    """
    features = {}
    # Recent 30-day prices
    recent_data = data.tail(30)
    features['recent_table'] = recent_data

    # Volume Chart using Plotly
    fig_volume = px.bar(data.reset_index(), x='Date', y='Volume', title="Volume Chart")
    features['volume_chart'] = fig_volume

    # 20-Day Moving Average Chart
    data['MA20'] = data['Close'].rolling(window=20).mean()
    fig_ma = go.Figure()
    fig_ma.add_trace(go.Scatter(x=data.index, y=data['MA20'], mode='lines', name='MA20', line=dict(color='red')))
    fig_ma.update_layout(title="20-Day Moving Average", xaxis_title="Date", yaxis_title="MA20")
    features['ma_chart'] = fig_ma

    # 20-Day Volatility Chart (rolling standard deviation)
    data['Volatility'] = data['Close'].rolling(window=20).std()
    fig_vol = go.Figure()
    fig_vol.add_trace(go.Scatter(x=data.index, y=data['Volatility'], mode='lines', name='Volatility', line=dict(color='orange')))
    fig_vol.update_layout(title="20-Day Volatility", xaxis_title="Date", yaxis_title="Volatility")
    features['vol_chart'] = fig_vol

    return features

def display_about():
    """
    Display about information in the sidebar.
    """
    st.sidebar.markdown("## About StockGPT")
    st.sidebar.info("""
    StockGPT is an advanced tool for analyzing and forecasting stock prices.
    It integrates historical data, news sentiment, and multiple forecasting models
    to provide actionable insights and interactive visualizations.
    """)

def display_feedback():
    """
    Display a feedback form in the sidebar.
    """
    st.sidebar.markdown("## Feedback")
    feedback = st.sidebar.text_area("Your Feedback:")
    if st.sidebar.button("Submit Feedback"):
        st.sidebar.success("Thank you for your feedback!")
        # Optionally log or store the feedback

# =============================================================================
# Main Application
# =============================================================================

def main():
    # Configure Streamlit page
    st.set_page_config(page_title="📈 Advanced StockGPT", layout="wide")
    display_about()
    display_feedback()
    
    # Sidebar input widgets
    ticker = st.sidebar.text_input("📌 Stock Ticker:", "AAPL").upper()
    start_date = st.sidebar.date_input("📅 Start Date", datetime.date.today() - datetime.timedelta(days=365))
    end_date = datetime.date.today()
    forecast_days = st.sidebar.slider("Forecast Days", 7, 60, 14)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Advanced Options")
    show_volume = st.sidebar.checkbox("Show Volume Chart", value=True)
    show_ma = st.sidebar.checkbox("Show Moving Average", value=True)
    show_volatility = st.sidebar.checkbox("Show Volatility", value=True)
    
    # Create multiple tabs for different sections
    tabs = st.tabs([
        "📊 Dashboard", 
        "📈 Charts", 
        "🕯️ Candlestick", 
        "🚀 Forecast", 
        "📰 News Impact", 
        "💡 Insights", 
        "📌 Detailed Analysis", 
        "⚙️ Settings"
    ])
    
    # ---------------------------
    # Data Acquisition
    # ---------------------------
    data_load_state = st.info("Fetching stock data...")
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            st.error("No data found for ticker. Please check the ticker symbol and try again.")
            return
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return
    data_load_state.success("Data fetched successfully!")
    data.index.name = "Date"  # Ensure the index is named for Plotly charts
    
    # ---------------------------
    # News and Sentiment Analysis
    # ---------------------------
    news_items = fetch_news(ticker)
    sentiment_score = sentiment_analysis(news_items)
    sentiment_factor = 1 + (sentiment_score * 0.05)
    
    # ---------------------------
    # Dashboard Tab: Overview
    # ---------------------------
    with tabs[0]:
        st.header(f"{ticker} Overview")
        try:
            closing_price = data['Close'].iloc[-1]
            if pd.isna(closing_price):
                closing_display = "N/A"
            else:
                try:
    closing_price = data['Close'].iloc[-1]
    if pd.isna(closing_price):
        closing_display = "N/A"
    else:
        closing_display = f"${float(closing_price):}"
except Exception as e:
    closing_display = "N/A"
    st.error(f"Error retrieving closing price: {e}")

st.metric("Today's Closing Price", closing_display)

        st.metric("Today's Closing Price", closing_display)
        st.metric("News Sentiment", f"{sentiment_score:.2f}")
        recommendation = "🟢 Buy" if sentiment_score > 0 else ("🔴 Hold/Sell" if sentiment_score < 0 else "⚪ Neutral")
        st.write("Investment Recommendation:", recommendation)
    
    # ---------------------------
    # Charts Tab: Historical Performance (Line Chart & Price Range Filter)
    # ---------------------------
    with tabs[1]:
        st.header("Historical Stock Performance")
        price_min = float(data['Close'].min())
        price_max = float(data['Close'].max())
        selected_range = st.slider("Select Closing Price Range", min_value=price_min, max_value=price_max, value=(price_min, price_max))
        
        filtered_data = data[(data['Close'] >= selected_range[0]) & (data['Close'] <= selected_range[1])]
        chart_data = filtered_data.reset_index()[['Date', 'Close']].dropna()
        
        if chart_data.empty:
            st.error("No chart data available for the selected price range.")
        else:
            try:
                fig_line = px.line(chart_data, x="Date", y="Close", title="Closing Prices Over Time", labels={"Close": "Closing Price"})
                st.plotly_chart(fig_line, use_container_width=True)
            except Exception as e:
                st.error(f"Error rendering chart: {e}")
        
        # Additional interactive features
        features = additional_interactive_features(data.copy())
        st.subheader("Recent Prices (Last 30 Days)")
        st.dataframe(features['recent_table'])
        if show_volume:
            st.subheader("Volume Chart")
            st.plotly_chart(features['volume_chart'], use_container_width=True)
        if show_ma:
            st.subheader("20-Day Moving Average")
            st.plotly_chart(features['ma_chart'], use_container_width=True)
        if show_volatility:
            st.subheader("20-Day Volatility")
            st.plotly_chart(features['vol_chart'], use_container_width=True)
    
    # ---------------------------
    # Candlestick Tab: Display Candlestick Chart
    # ---------------------------
    with tabs[2]:
        st.header("Candlestick Chart")
        try:
            fig_candle = go.Figure(data=[go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name=ticker
            )])
            fig_candle.update_layout(title=f"{ticker} Candlestick Chart", xaxis_title="Date", yaxis_title="Price")
            st.plotly_chart(fig_candle, use_container_width=True)
        except Exception as e:
            st.error(f"Error rendering candlestick chart: {e}")
    
    # ---------------------------
    # Forecast Tab: Price Forecasting Using Multiple Models
    # ---------------------------
    with tabs[3]:
        st.header("Stock Price Forecast")
        st.write("Forecasting using Prophet, ARIMA, and LSTM models.")
        try:
            prophet_pred = forecast_prophet(data, forecast_days)
        except Exception as e:
            st.error(f"Prophet forecasting failed: {e}")
            prophet_pred = np.zeros(forecast_days)
        try:
            arima_pred = forecast_arima(data['Close'], forecast_days)
        except Exception as e:
            st.error(f"ARIMA forecasting failed: {e}")
            arima_pred = np.zeros(forecast_days)
        try:
            lstm_pred = forecast_lstm(data['Close'], forecast_days)
        except Exception as e:
            st.error(f"LSTM forecasting failed: {e}")
            lstm_pred = np.zeros(forecast_days)
        
        if len(data['Close']) >= forecast_days:
            actual_recent = data['Close'][-forecast_days:].values
        else:
            actual_recent = prophet_pred
        
        errors = {
            "Prophet": mean_absolute_error(actual_recent, prophet_pred),
            "ARIMA": mean_absolute_error(actual_recent, arima_pred),
            "LSTM": mean_absolute_error(actual_recent, lstm_pred)
        }
        best_model = min(errors, key=errors.get)
        best_forecast = {"Prophet": prophet_pred, "ARIMA": arima_pred, "LSTM": lstm_pred}[best_model]
        
        adjusted_forecast = best_forecast * sentiment_factor
        
        forecast_dates = pd.date_range(start=end_date, periods=forecast_days+1)[1:]
        forecast_df = pd.DataFrame({
            "Date": forecast_dates,
            "Forecasted Price": best_forecast.round(2),
            "Adjusted Forecast Price": adjusted_forecast.round(2)
        })
        
        st.success(f"Best Forecast Model: **{best_model}** with MAE: {errors[best_model]:.2f}")
        st.dataframe(forecast_df.style.format({
            "Forecasted Price": "${:,.2f}", 
            "Adjusted Forecast Price": "${:,.2f}"
        }))
        
        forecast_chart_data = forecast_df.melt(id_vars="Date", value_vars=["Forecasted Price", "Adjusted Forecast Price"], var_name="Type", value_name="Price")
        try:
            fig_forecast = px.line(forecast_chart_data, x="Date", y="Price", color="Type", title="Forecast Comparison")
            st.plotly_chart(fig_forecast, use_container_width=True)
        except Exception as e:
            st.error(f"Error rendering forecast chart: {e}")
    
    # ---------------------------
    # News Impact Tab: Summaries of Relevant News
    # ---------------------------
    with tabs[4]:
        st.header("News Summaries Impacting the Stock")
        news_df = get_news_summaries(news_items)
        if not news_df.empty:
            for idx, row in news_df.iterrows():
                st.subheader(row['Title'])
                st.write(row['Summary'])
                st.markdown("---")
        else:
            st.write("No news items available.")
    
    # ---------------------------
    # Insights Tab: Analysis, Recommendations & Interactive Q&A
    # ---------------------------
    with tabs[5]:
        st.header("Insights & Recommendations")
        st.markdown("""
        **Market Analysis:**
        - Positive sentiment usually indicates potential upward momentum.
        - Negative sentiment can be a warning sign of downturns.
        - Always consider multiple market indicators before making investment decisions.
        
        **Recommendations:**
        - If sentiment is positive: Consider buying and holding.
        - If sentiment is negative: Exercise caution; consider selling or holding.
        """)
        st.markdown("### Ask a Question")
        question = st.text_input("Enter your question about market trends or stock performance:")
        if st.button("Get Answer"):
            if "increase" in question.lower():
                st.write("Stocks may increase if there is sustained positive sentiment and strong earnings reports.")
            elif "decrease" in question.lower():
                st.write("Stocks might decrease if negative news continues and market conditions worsen.")
            else:
                st.write("Please provide more details or ask another question.")
    
    # ---------------------------
    # Detailed Analysis Tab: In-Depth Data Exploration
    # ---------------------------
    with tabs[6]:
        st.header("Detailed Data Analysis")
        st.markdown("Explore various aspects of the stock data.")
        analysis_start = st.date_input("Analysis Start Date", start_date)
        analysis_end = st.date_input("Analysis End Date", end_date)
        if analysis_start > analysis_end:
            st.error("Start date must be before end date.")
        else:
            detailed_data = data.loc[analysis_start:analysis_end]
            st.write("Detailed Data", detailed_data)
            st.subheader("Correlation Matrix")
            corr = detailed_data.corr()
            st.dataframe(corr.style.background_gradient(cmap='coolwarm'))
            st.subheader("Distribution of Closing Prices")
            try:
                fig_hist = px.histogram(detailed_data.reset_index(), x="Close", nbins=30, title="Distribution of Closing Prices")
                st.plotly_chart(fig_hist, use_container_width=True)
            except Exception as e:
                st.error(f"Error rendering histogram: {e}")
    
    # ---------------------------
    # Settings Tab: Application Options
    # ---------------------------
    with tabs[7]:
        st.header("Application Settings")
        st.markdown("Adjust application parameters and view raw data.")
        if st.checkbox("Show raw data"):
            st.subheader("Raw Data")
            st.dataframe(data)
        st.markdown("### Model Settings")
        st.markdown("Forecasting model parameters can be adjusted here in future versions.")
    
    # ---------------------------
    # Footer and Additional Sections
    # ---------------------------
    st.markdown("---")
    st.write("© 2025 Advanced StockGPT - All rights reserved.")
    st.markdown("### Future Enhancements")
    st.write("More features and interactive elements will be added in future updates.")
    st.markdown("### End of Dashboard")

# =============================================================================
# Run the Application
# =============================================================================
if __name__ == "__main__":
    main()
    logging.info("Application started.")
