# design.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import plotly.express as px
import plotly.graph_objects as go
import logging
import warnings

from forecast_models import forecast_prophet, forecast_arima, forecast_lstm
from nlp_utils import fetch_news, sentiment_analysis, get_news_summaries
from additional_factors import calculate_technical_indicators
from model_tuning import tune_prophet, tune_arima, tune_lstm

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def additional_interactive_features(data):
    features = {}
    features['recent_table'] = data.tail(30)
    
    # 20-Day Moving Average Chart
    data['MA20'] = data['Close'].rolling(window=20).mean()
    fig_ma = go.Figure()
    fig_ma.add_trace(go.Scatter(x=data.index, y=data['MA20'], mode='lines', name='MA20', line=dict(color='red')))
    fig_ma.update_layout(title="20-Day Moving Average", xaxis_title="Date", yaxis_title="MA20")
    features['ma_chart'] = fig_ma

    # 20-Day Volatility Chart
    data['Volatility'] = data['Close'].rolling(window=20).std()
    fig_vol = go.Figure()
    fig_vol.add_trace(go.Scatter(x=data.index, y=data['Volatility'], mode='lines', name='Volatility', line=dict(color='orange')))
    fig_vol.update_layout(title="20-Day Volatility", xaxis_title="Date", yaxis_title="Volatility")
    features['vol_chart'] = fig_vol

    return features

def display_about():
    st.sidebar.markdown("## About StockGPT")
    st.sidebar.info(
        "StockGPT is a comprehensive stock analysis and forecasting tool. "
        "It integrates historical data, extended news sentiment (economic, political, and company-specific), "
        "technical indicators, and multiple forecasting models with hyper-parameter tuning to deliver actionable insights."
    )

def display_feedback():
    st.sidebar.markdown("## Feedback")
    feedback = st.sidebar.text_area("Your Feedback:")
    if st.sidebar.button("Submit Feedback"):
        st.sidebar.success("Thank you for your feedback!")

def flatten_data_columns(data):
    """
    Flatten multi-index columns from yfinance data if present.
    """
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data

def main():
    st.set_page_config(page_title="📈 Advanced StockGPT", layout="wide")
    display_about()
    display_feedback()
    
    # Sidebar inputs
    ticker = st.sidebar.text_input("📌 Stock Ticker:", "WMT").upper()
    start_date = st.sidebar.date_input("📅 Start Date", datetime.date.today() - datetime.timedelta(days=365))
    end_date = datetime.date.today()
    forecast_days = st.sidebar.slider("Forecast Days", 7, 60, 14)
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Advanced Options")
    show_ma = st.sidebar.checkbox("Show Moving Average", value=True)
    show_volatility = st.sidebar.checkbox("Show Volatility", value=True)
    
    # Tabs
    tabs = st.tabs(["📊 Dashboard", "📈 Charts", "🕯️ Candlestick", "🚀 Forecast", "📰 News Impact", "💡 Insights", "📌 Detailed Analysis", "⚙️ Settings"])
    
    # Data Acquisition
    data_load_state = st.info("Fetching stock data...")
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            st.error("No data found for ticker. Please check the ticker symbol and try again.")
            return
        data = flatten_data_columns(data)
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return
    data_load_state.success("Data fetched successfully!")
    data.index.name = "Date"
    
    # Calculate technical indicators
    data = calculate_technical_indicators(data)
    
    # News & Sentiment Analysis
    news_items = fetch_news(ticker)
    sentiment_score = sentiment_analysis(news_items)
    sentiment_factor = 1 + (sentiment_score * 0.05)
    
    with tabs[0]:
        st.header(f"{ticker} Overview")
        try:
            closing_price = data['Close'].iloc[-1]
            closing_display = f"${closing_price:.2f}" if not pd.isna(closing_price) else "N/A"
        except Exception as e:
            closing_display = "N/A"
            st.error(f"Error retrieving closing price: {e}")
        st.metric("Today's Closing Price", closing_display)
        st.metric("News Sentiment", f"{sentiment_score:.2f}")
        recommendation = "🟢 Buy" if sentiment_score > 0 else ("🔴 Hold/Sell" if sentiment_score < 0 else "⚪ Neutral")
        st.write("Investment Recommendation:", recommendation)
    
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
        features = additional_interactive_features(data.copy())
        st.subheader("Recent Prices (Last 30 Days)")
        st.dataframe(features['recent_table'])
        if show_ma:
            st.subheader("20-Day Moving Average")
            st.plotly_chart(features['ma_chart'], use_container_width=True)
        if show_volatility:
            st.subheader("20-Day Volatility")
            st.plotly_chart(features['vol_chart'], use_container_width=True)
    
    with tabs[2]:
        st.header("Candlestick Chart")
        try:
            candle_data = data.reset_index()
            fig_candle = go.Figure(data=[go.Candlestick(
                x=candle_data['Date'],
                open=candle_data['Open'],
                high=candle_data['High'],
                low=candle_data['Low'],
                close=candle_data['Close'],
                name=ticker
            )])
            fig_candle.update_layout(title=f"{ticker} Candlestick Chart", xaxis_title="Date", yaxis_title="Price")
            st.plotly_chart(fig_candle, use_container_width=True)
        except Exception as e:
            st.error(f"Error rendering candlestick chart: {e}")
    
    with tabs[3]:
        st.header("Stock Price Forecast")
        st.write("Forecasting using Prophet, ARIMA, and LSTM models with hyper-parameter tuning in the background.")
        
        # Tuning models (simulated)
        prophet_params = tune_prophet(data)
        arima_params = tune_arima(data['Close'])
        lstm_params = tune_lstm(data['Close'])
        
        try:
            prophet_result = forecast_prophet(data, forecast_days, tuned_params=prophet_params)
        except Exception as e:
            st.error(f"Prophet forecasting failed: {e}")
            prophet_result = pd.DataFrame({"forecast": np.zeros(forecast_days),
                                           "lower": np.zeros(forecast_days),
                                           "upper": np.zeros(forecast_days)})
        try:
            arima_result = forecast_arima(data['Close'], forecast_days, tuned_params=arima_params)
        except Exception as e:
            st.error(f"ARIMA forecasting failed: {e}")
            arima_result = pd.DataFrame({"forecast": np.zeros(forecast_days),
                                         "lower": np.zeros(forecast_days),
                                         "upper": np.zeros(forecast_days)})
        try:
            lstm_result = forecast_lstm(data['Close'], forecast_days, tuned_params=lstm_params)
        except Exception as e:
            st.error(f"LSTM forecasting failed: {e}")
            lstm_result = pd.DataFrame({"forecast": np.zeros(forecast_days),
                                        "lower": np.zeros(forecast_days),
                                        "upper": np.zeros(forecast_days)})
        
        if len(data['Close']) >= forecast_days:
            actual_recent = data['Close'][-forecast_days:].values
        else:
            actual_recent = prophet_result["forecast"].values
        
        errors = {
            "Prophet": np.abs(actual_recent - prophet_result["forecast"].values).mean(),
            "ARIMA": np.abs(actual_recent - arima_result["forecast"].values).mean(),
            "LSTM": np.abs(actual_recent - lstm_result["forecast"].values).mean()
        }
        best_model = min(errors, key=errors.get)
        best_result = prophet_result if best_model=="Prophet" else (arima_result if best_model=="ARIMA" else lstm_result)
        
        # Apply sentiment adjustment to forecast and confidence intervals
        best_result_adj = best_result.copy()
        best_result_adj["forecast"] *= sentiment_factor
        best_result_adj["lower"] *= sentiment_factor
        best_result_adj["upper"] *= sentiment_factor
        
        forecast_dates = pd.date_range(start=end_date, periods=forecast_days+1)[1:]
        forecast_df = best_result_adj.copy()
        forecast_df["Date"] = forecast_dates
        st.success(f"Best Forecast Model: **{best_model}** with simulated MAE: {errors[best_model]:.2f}")
        st.dataframe(forecast_df.style.format({
            "forecast": "${:,.2f}",
            "lower": "${:,.2f}",
            "upper": "${:,.2f}"
        }))
        
        forecast_chart_data = forecast_df.melt(id_vars="Date", value_vars=["forecast", "lower", "upper"],
                                               var_name="Type", value_name="Price")
        try:
            fig_forecast = px.line(forecast_chart_data, x="Date", y="Price", color="Type", title="Forecast Comparison with 95% CI")
            st.plotly_chart(fig_forecast, use_container_width=True)
        except Exception as e:
            st.error(f"Error rendering forecast chart: {e}")
    
    with tabs[4]:
        st.header("News Summaries Impacting the Stock")
        news_df = get_news_summaries(fetch_news(ticker))
        if not news_df.empty:
            for idx, row in news_df.iterrows():
                st.subheader(row['Title'])
                st.write(row['Summary'])
                st.markdown("---")
        else:
            st.write("No news items available.")
    
    with tabs[5]:
        st.header("Insights & Recommendations")
        st.markdown("""
        **Market Analysis:**
        - Positive sentiment indicates potential upward momentum.
        - Negative sentiment serves as a warning.
        - Technical indicators (RSI, MACD) and economic/political news provide additional context.
        
        **Recommendations:**
        - Favor buying when sentiment is positive and indicators are supportive.
        - Exercise caution or consider selling when sentiment is negative or indicators are overbought.
        """)
        st.markdown("### Ask a Question")
        question = st.text_input("Enter your question about market trends or stock performance:")
        if st.button("Get Answer"):
            if "increase" in question.lower():
                st.write("Stocks may increase if sustained positive sentiment, strong earnings, and favorable technical indicators contin
