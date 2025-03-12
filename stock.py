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


def flatten_data_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Flatten multi-index columns from yfinance data if present."""
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data

def additional_interactive_features(data: pd.DataFrame):
    """Generate interactive charts (MA, Volatility, RSI, MACD, etc.) and recent data table."""
    features = {}

    # Keep a copy for calculations
    data_calc = data.copy()

    # Recent 30-day prices
    features['recent_table'] = data_calc.tail(30).round(2)
    
    # 20-Day Moving Average
    data_calc['MA20'] = data_calc['Close'].rolling(window=20).mean()
    fig_ma = go.Figure()
    fig_ma.add_trace(go.Scatter(x=data_calc.index, y=data_calc['MA20'], mode='lines', name='MA20', line=dict(color='red')))
    fig_ma.update_layout(title="20-Day Moving Average", xaxis_title="Date", yaxis_title="MA20")
    features['ma_chart'] = fig_ma

    # 20-Day Volatility
    data_calc['Volatility'] = data_calc['Close'].rolling(window=20).std()
    fig_vol = go.Figure()
    fig_vol.add_trace(go.Scatter(x=data_calc.index, y=data_calc['Volatility'], mode='lines', name='Volatility', line=dict(color='orange')))
    fig_vol.update_layout(title="20-Day Volatility", xaxis_title="Date", yaxis_title="Volatility")
    features['vol_chart'] = fig_vol

    # RSI
    if 'RSI' in data_calc.columns:
        fig_rsi = px.line(data_calc.reset_index(), x="Date", y="RSI", title="RSI Over Time")
        features['rsi_chart'] = fig_rsi

    # MACD & Signal
    if 'MACD' in data_calc.columns and 'MACD_Signal' in data_calc.columns:
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=data_calc.index, y=data_calc['MACD'], mode='lines', name='MACD'))
        fig_macd.add_trace(go.Scatter(x=data_calc.index, y=data_calc['MACD_Signal'], mode='lines', name='Signal'))
        fig_macd.update_layout(title="MACD & Signal", xaxis_title="Date", yaxis_title="MACD")
        features['macd_chart'] = fig_macd

    # MACD Histogram
    if 'MACD_Hist' in data_calc.columns:
        fig_hist = px.bar(data_calc.reset_index(), x="Date", y="MACD_Hist", title="MACD Histogram")
        features['macd_hist_chart'] = fig_hist

    return features


def display_about():
    """Sidebar about section."""
    st.sidebar.markdown("## About StockGPT")
    st.sidebar.info(
        "StockGPT is a comprehensive stock analysis and forecasting tool. "
        "It integrates historical data, extended news sentiment (including economic and political news), "
        "technical indicators (RSI, MACD), and multiple forecasting models with hyper-parameter tuning to deliver actionable insights."
    )

def display_feedback():
    """Sidebar feedback section."""
    st.sidebar.markdown("## Feedback")
    feedback = st.sidebar.text_area("Your Feedback:")
    if st.sidebar.button("Submit Feedback"):
        st.sidebar.success("Thank you for your feedback!")

def main():
    st.set_page_config(page_title="📈 Advanced StockGPT", layout="wide")
    display_about()
    display_feedback()
    
    # Sidebar inputs
    ticker = st.sidebar.text_input("📌 Stock Ticker:", "AAPL").upper()
    start_date = st.sidebar.date_input("📅 Start Date", datetime.date.today() - datetime.timedelta(days=365))
    end_date = datetime.date.today()
    forecast_days = st.sidebar.slider("Forecast Days", 7, 60, 14)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Advanced Options")
    show_ma = st.sidebar.checkbox("Show Moving Average", value=True)
    show_volatility = st.sidebar.checkbox("Show Volatility", value=True)
    
    # Create tabs
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
    
    # Fetch Data
    data_load_state = st.info("Fetching stock data...")
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            st.error("No data found for this ticker. Please check the symbol and try again.")
            return
        data = flatten_data_columns(data)
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return
    data_load_state.success("Data fetched successfully!")
    data.index.name = "Date"
    
    # Compute technical indicators
    data = calculate_technical_indicators(data)
    
    # Fetch news & compute sentiment
    news_items = fetch_news(ticker)
    sentiment_score = sentiment_analysis(news_items)
    sentiment_factor = 1 + (sentiment_score * 0.05)
    
    # ================
    #  TAB: Dashboard
    # ================
    with tabs[0]:
        # Mimic the Apple Inc style page
        st.title(f"{ticker} - Overview Page")
        
        col1, col2 = st.columns([3, 1], gap="large")
        
        with col1:
            st.subheader(f"NASDAQ: {ticker}")
            try:
                current_price = data['Close'].iloc[-1]
                prev_price = data['Close'].iloc[-2] if len(data) > 1 else current_price
                change = current_price - prev_price
                pct_change = (change / prev_price) * 100 if prev_price != 0 else 0
                color_style = "red" if change < 0 else "green"
                
                st.markdown(
                    f"""
                    <div style='font-size: 2rem; font-weight: bold;'>
                        ${current_price:.2f}
                        <span style='color: {color_style}; font-size: 1.5rem;'>
                            {change:+.2f} ({pct_change:+.2f}%)
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.caption(f"As of {datetime.datetime.now().strftime('%B %d, %Y %I:%M %p')} EDT")
            except Exception as e:
                st.error(f"Error retrieving price: {e}")
            
            # Small line chart for recent close
            try:
                last_10 = data['Close'].tail(10)
                st.line_chart(last_10, height=200)
            except Exception as e:
                st.error(f"Error rendering mini chart: {e}")
        
        with col2:
            st.markdown("### Explore more")
            # Dummy watchlist or related companies
            st.write("**Microsoft Corp (MSFT)** +0.78%")
            st.write("**Samsung Electronics** +0.29%")
            st.write("**Amazon Inc (AMZN)** +0.88%")
            st.write("**NVIDIA Corp (NVDA)** +1.66%")
        
        st.markdown("---")
        
        # Company Info (you can expand or fetch real data)
        st.subheader(f"{ticker} Company Info")
        st.write(
            f"{ticker} is an American multinational technology company headquartered in Cupertino, "
            "California, United States. It designs, develops, and sells consumer electronics, "
            "computer software, and online services."
        )
        
        col3, col4 = st.columns(2)
        with col3:
            st.write("**CEO:** Tim Cook")
            st.write("**Founded:** April 1, 1976 (by Steve Jobs, Steve Wozniak, Ronald Wayne)")
            st.write("**Headquarters:** Cupertino, California, United States")
        with col4:
            st.write("**Revenue:** $365.8B (2021)")
            st.write("**Employees:** ~164,000 (2022)")
        
        st.markdown("---")
        st.subheader("Related News")
        news_df = get_news_summaries(news_items)
        if not news_df.empty:
            for idx, row in news_df.iterrows():
                with st.expander(row['Title']):
                    st.write(row['Summary'])
        else:
            st.write("No news items available.")
    
    # =================
    #  TAB: Charts
    # =================
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
        if 'rsi_chart' in features:
            st.subheader("RSI Chart")
            st.plotly_chart(features['rsi_chart'], use_container_width=True)
        if 'macd_chart' in features:
            st.subheader("MACD & Signal")
            st.plotly_chart(features['macd_chart'], use_container_width=True)
        if 'macd_hist_chart' in features:
            st.subheader("MACD Histogram")
            st.plotly_chart(features['macd_hist_chart'], use_container_width=True)
    
    # =================
    #  TAB: Candlestick
    # =================
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
    
    # =================
    #  TAB: Forecast
    # =================
    with tabs[3]:
        st.header("Stock Price Forecast")
        st.write("Forecasting using Prophet, ARIMA, and LSTM models (with hyper-parameter tuning).")
        
        # Tuning (simulated)
        prophet_params = tune_prophet(data)
        arima_params = tune_arima(data['Close'])
        lstm_params = tune_lstm(data['Close'])
        
        try:
            prophet_result = forecast_prophet(data, forecast_days, tuned_params=prophet_params)
        except Exception as e:
            st.error(f"Prophet forecasting failed: {e}")
            prophet_result = pd.DataFrame({"forecast": np.zeros(forecast_days), "lower": np.zeros(forecast_days), "upper": np.zeros(forecast_days)})
        
        try:
            arima_result = forecast_arima(data['Close'], forecast_days, tuned_params=arima_params)
        except Exception as e:
            st.error(f"ARIMA forecasting failed: {e}")
            arima_result = pd.DataFrame({"forecast": np.zeros(forecast_days), "lower": np.zeros(forecast_days), "upper": np.zeros(forecast_days)})
        
        try:
            lstm_result = forecast_lstm(data['Close'], forecast_days, tuned_params=lstm_params)
        except Exception as e:
            st.error(f"LSTM forecasting failed: {e}")
            lstm_result = pd.DataFrame({"forecast": np.zeros(forecast_days), "lower": np.zeros(forecast_days), "upper": np.zeros(forecast_days)})
        
        # Evaluate which model is best
        if len(data['Close']) >= forecast_days:
            actual_recent = data['Close'][-forecast_days:].values
        else:
            actual_recent = prophet_result["forecast"].values  # fallback
        
        errors = {
            "Prophet": np.abs(actual_recent - prophet_result["forecast"].values).mean(),
            "ARIMA": np.abs(actual_recent - arima_result["forecast"].values).mean(),
            "LSTM": np.abs(actual_recent - lstm_result["forecast"].values).mean()
        }
        best_model = min(errors, key=errors.get)
        best_result = prophet_result if best_model=="Prophet" else (arima_result if best_model=="ARIMA" else lstm_result)
        
        # Apply sentiment factor
        best_result_adj = best_result.copy()
        best_result_adj["forecast"] *= sentiment_factor
        best_result_adj["lower"] *= sentiment_factor
        best_result_adj["upper"] *= sentiment_factor
        
        forecast_dates = pd.date_range(start=end_date, periods=forecast_days+1)[1:]
        forecast_df = best_result_adj.copy()
        forecast_df["Date"] = forecast_dates
        st.success(f"Best Forecast Model: **{best_model}** | MAE: {errors[best_model]:.2f}")
        st.dataframe(forecast_df.round(2).style.format({"forecast": "${:,.2f}", "lower": "${:,.2f}", "upper": "${:,.2f}"}))
        
        # Chart
        forecast_chart_data = forecast_df.melt(id_vars="Date", value_vars=["forecast", "lower", "upper"], var_name="Type", value_name="Price")
        try:
            fig_forecast = px.line(forecast_chart_data, x="Date", y="Price", color="Type", title="Forecast Comparison (95% CI)")
            st.plotly_chart(fig_forecast, use_container_width=True)
        except Exception as e:
            st.error(f"Error rendering forecast chart: {e}")
    
    # =================
    #  TAB: News Impact
    # =================
    with tabs[4]:
        st.header("News Summaries Impacting the Stock")
        # We already fetched news_items above
        news_df = get_news_summaries(news_items)
        if not news_df.empty:
            for idx, row in news_df.iterrows():
                st.subheader(row['Title'])
                st.write(row['Summary'])
                st.markdown("---")
        else:
            st.write("No news items available.")
    
    # =================
    #  TAB: Insights
    # =================
    with tabs[5]:
        st.header("Insights & Recommendations")
        st.markdown("""
        **Market Analysis:**
        - Positive sentiment indicates potential upward momentum.
        - Negative sentiment can be a warning sign.
        - Technical indicators (RSI, MACD) provide more context.
        - Economic and political news can sway investor sentiment.

        **Recommendations:**
        - Consider buying if sentiment is positive and indicators are strong.
        - If sentiment is negative, consider selling or holding.
        """)
        
        st.markdown("### Ask a Question")
        question = st.text_input("Enter your question about market trends or stock performance:")
        if st.button("Get Answer"):
            if "increase" in question.lower():
                st.write("Stocks may increase if sustained positive sentiment, strong earnings, and favorable technical indicators continue.")
            elif "decrease" in question.lower():
                st.write("Stocks might decrease if negative news and bearish technical indicators persist.")
            else:
                st.write("Please provide more details or ask another question.")
    
    # ========================
    #  TAB: Detailed Analysis
    # ========================
    with tabs[6]:
        st.header("Detailed Data Analysis")
        st.markdown("Explore various aspects of the stock data.")
        analysis_start = st.date_input("Analysis Start Date", start_date)
        analysis_end = st.date_input("Analysis End Date", end_date)
        if analysis_start > analysis_end:
            st.error("Start date must be before end date.")
        else:
            detailed_data = data.loc[analysis_start:analysis_end]
            st.write("Detailed Data", detailed_data.round(2))
            st.subheader("Correlation Matrix")
            corr = detailed_data.corr().round(2)
            st.dataframe(corr.style.background_gradient(cmap='coolwarm'))
            st.subheader("Distribution of Closing Prices")
            try:
                fig_hist = px.histogram(detailed_data.reset_index(), x="Close", nbins=30, title="Distribution of Closing Prices")
                st.plotly_chart(fig_hist, use_container_width=True)
            except Exception as e:
                st.error(f"Error rendering histogram: {e}")
    
    # ===============
    #  TAB: Settings
    # ===============
    with tabs[7]:
        st.header("Application Settings")
        st.markdown("Adjust application parameters and view raw data.")
        if st.checkbox("Show raw data"):
            st.subheader("Raw Data")
            st.dataframe(data.round(2))
        st.markdown("### Model Settings")
        st.markdown("Forecasting model parameters can be adjusted here in future versions.")

if __name__ == "__main__":
    main()
