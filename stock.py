# design.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import plotly.graph_objects as go
import plotly.express as px
import logging
import warnings

from forecast_models import forecast_prophet, forecast_arima, forecast_lstm
from nlp_utils import fetch_news, sentiment_analysis, get_news_summaries
from additional_factors import calculate_technical_indicators
from model_tuning import tune_prophet, tune_arima, tune_lstm

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Company information mapping for demonstration
COMPANY_INFO = {
    "AAPL": {
        "Name": "Apple Inc.",
        "CEO": "Tim Cook",
        "Founded": "April 1, 1976 (by Steve Jobs, Steve Wozniak, Ronald Wayne)",
        "Headquarters": "Cupertino, California, USA",
        "Revenue": "$365.8B (2021)",
        "Employees": "~164,000 (2022)",
        "Description": "Apple Inc. is an American multinational technology company that designs, develops, and sells consumer electronics, computer software, and online services."
    },
    "MSFT": {
        "Name": "Microsoft Corporation",
        "CEO": "Satya Nadella",
        "Founded": "April 4, 1975 (by Bill Gates and Paul Allen)",
        "Headquarters": "Redmond, Washington, USA",
        "Revenue": "$168.1B (2021)",
        "Employees": "~181,000 (2022)",
        "Description": "Microsoft Corporation is a global technology company that produces computer software, consumer electronics, personal computers, and related services."
    },
    # Add more companies as needed...
}

def flatten_data_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Flatten multi-index columns from yfinance data if present."""
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data

def additional_interactive_features(data: pd.DataFrame):
    """Generate additional charts and recent table."""
    features = {}
    data_calc = data.copy()

    # Recent 30-day prices table (rounded)
    features['recent_table'] = data_calc.tail(30).round(2)
    
    # 20-Day Moving Average Chart
    data_calc['MA20'] = data_calc['Close'].rolling(window=20).mean()
    fig_ma = go.Figure()
    fig_ma.add_trace(go.Scatter(x=data_calc.index, y=data_calc['MA20'], mode='lines', name='MA20', line=dict(color='red')))
    fig_ma.update_layout(title="20-Day Moving Average", xaxis_title="Date", yaxis_title="MA20")
    features['ma_chart'] = fig_ma

    # 20-Day Volatility Chart
    data_calc['Volatility'] = data_calc['Close'].rolling(window=20).std()
    fig_vol = go.Figure()
    fig_vol.add_trace(go.Scatter(x=data_calc.index, y=data_calc['Volatility'], mode='lines', name='Volatility', line=dict(color='orange')))
    fig_vol.update_layout(title="20-Day Volatility", xaxis_title="Date", yaxis_title="Volatility")
    features['vol_chart'] = fig_vol

    # RSI Chart if available
    if 'RSI' in data_calc.columns:
        fig_rsi = px.line(data_calc.reset_index(), x="Date", y="RSI", title="RSI Over Time")
        features['rsi_chart'] = fig_rsi

    # MACD & Signal Chart if available
    if 'MACD' in data_calc.columns and 'MACD_Signal' in data_calc.columns:
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=data_calc.index, y=data_calc['MACD'], mode='lines', name='MACD'))
        fig_macd.add_trace(go.Scatter(x=data_calc.index, y=data_calc['MACD_Signal'], mode='lines', name='Signal'))
        fig_macd.update_layout(title="MACD & Signal", xaxis_title="Date", yaxis_title="MACD")
        features['macd_chart'] = fig_macd

    # MACD Histogram Chart if available
    if 'MACD_Hist' in data_calc.columns:
        fig_hist = px.bar(data_calc.reset_index(), x="Date", y="MACD_Hist", title="MACD Histogram")
        features['macd_hist_chart'] = fig_hist

    return features

def display_about():
    st.sidebar.markdown("## About StockGPT")
    st.sidebar.info(
        "StockGPT is a cutting-edge stock analysis and forecasting tool. It integrates historical data, "
        "news sentiment (economic, political, and company-specific), technical indicators, and multiple forecasting models "
        "with hyper-parameter tuning to provide deep insights and actionable recommendations."
    )

def display_feedback():
    st.sidebar.markdown("## Feedback")
    feedback = st.sidebar.text_area("Your Feedback:")
    if st.sidebar.button("Submit Feedback"):
        st.sidebar.success("Thank you for your feedback!")

def combine_historical_and_forecast(data: pd.DataFrame, forecast_df: pd.DataFrame, hist_start: datetime.date, hist_end: datetime.date) -> pd.DataFrame:
    """
    Combine historical data with forecast data into one DataFrame for plotting.
    """
    hist_data = data.reset_index()[['Date', 'Close']].copy()
    hist_data = hist_data[(hist_data['Date'] >= pd.to_datetime(hist_start)) & (hist_data['Date'] <= pd.to_datetime(hist_end))]
    hist_data['Type'] = 'Historical'
    hist_data.rename(columns={'Close': 'Price'}, inplace=True)
    
    fc_data = forecast_df[['Date', 'forecast']].copy()
    fc_data['Type'] = 'Forecast'
    fc_data.rename(columns={'forecast': 'Price'}, inplace=True)
    
    combined = pd.concat([hist_data, fc_data], ignore_index=True)
    return combined

def main():
    st.set_page_config(page_title="📈 Advanced StockGPT", layout="wide")
    display_about()
    display_feedback()
    
    # Sidebar inputs
    ticker = st.sidebar.text_input("📌 Stock Ticker:", "AAPL").upper()
    start_date = st.sidebar.date_input("📅 Start Date", datetime.date.today() - datetime.timedelta(days=365))
    end_date = datetime.date.today()
    forecast_days = st.sidebar.slider("Forecast Days", 7, 365, 30, help="Number of days to forecast")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Advanced Options")
    show_ma = st.sidebar.checkbox("Show Moving Average", value=True)
    show_volatility = st.sidebar.checkbox("Show Volatility", value=True)
    
    # Tabs (remove the separate Candlestick tab)
    tabs = st.tabs([
        "📊 Dashboard", 
        "📈 Charts", 
        "🚀 Forecast", 
        "📰 News Impact", 
        "💡 Insights", 
        "📌 Detailed Analysis", 
        "⚙️ Settings"
    ])
    
    # Fetch data
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
    
    # Compute technical indicators (RSI, MACD, etc.)
    data = calculate_technical_indicators(data)
    
    # News & sentiment
    news_items = fetch_news(ticker)
    sentiment_score = sentiment_analysis(news_items)
    sentiment_factor = 1 + (sentiment_score * 0.05)
    
    # -------------------
    # TAB: Dashboard
    # -------------------
    with tabs[0]:
        # Use custom CSS for styling
        st.markdown(
            """
            <style>
            .price-container {
                background-color: #f0f2f6;
                border-radius: 8px;
                padding: 1rem;
                margin-bottom: 1rem;
            }
            .price-text {
                font-size: 2rem;
                font-weight: bold;
            }
            .change-text {
                font-size: 1.2rem;
                margin-left: 1rem;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        
        st.title(f"{ticker} - Company Overview")
        
        col1, col2 = st.columns([3, 1], gap="large")
        with col1:
            # Candlestick chart embedded in Dashboard
            try:
                candle_data = data.reset_index()
                fig_candle = go.Figure(data=[go.Candlestick(
                    x=candle_data['Date'],
                    open=candle_data['Open'].round(2),
                    high=candle_data['High'].round(2),
                    low=candle_data['Low'].round(2),
                    close=candle_data['Close'].round(2),
                    name=ticker
                )])
                fig_candle.update_layout(title=f"{ticker} Candlestick Chart", xaxis_title="Date", yaxis_title="Price")
                st.plotly_chart(fig_candle, use_container_width=True)
            except Exception as e:
                st.error(f"Error rendering candlestick chart: {e}")
            
            # Display key metrics (price and change)
            try:
                current_price = data['Close'].iloc[-1]
                prev_price = data['Close'].iloc[-2] if len(data) > 1 else current_price
                change = current_price - prev_price
                pct_change = (change / prev_price * 100) if prev_price != 0 else 0
                color = "green" if change >= 0 else "red"
                st.markdown(
                    f"""
                    <div class='price-container'>
                        <span class='price-text'>${current_price:.2f}</span>
                        <span class='change-text' style='color:{color};'>({change:+.2f}, {pct_change:+.2f}%)</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            except Exception as e:
                st.error(f"Error retrieving price: {e}")
            
            # Historical + Forecast combined chart option
            show_forecast_chart = st.checkbox("Show 1-Year Forecast Overlay", value=False)
            if show_forecast_chart:
                # Use tuning and forecasting to get 365-day forecast
                prophet_params = tune_prophet(data)
                arima_params = tune_arima(data['Close'])
                lstm_params = tune_lstm(data['Close'])
                try:
                    prophet_result = forecast_prophet(data, 365, tuned_params=prophet_params)
                except:
                    prophet_result = pd.DataFrame({"forecast": np.zeros(365)})
                try:
                    arima_result = forecast_arima(data['Close'], 365, tuned_params=arima_params)
                except:
                    arima_result = pd.DataFrame({"forecast": np.zeros(365)})
                try:
                    lstm_result = forecast_lstm(data['Close'], 365, tuned_params=lstm_params)
                except:
                    lstm_result = pd.DataFrame({"forecast": np.zeros(365)})
                
                # Select best model (here, just choose Prophet for demo)
                best_forecast = prophet_result.copy()
                best_forecast["forecast"] = best_forecast["forecast"].round(2) * sentiment_factor
                forecast_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=365)
                best_forecast["Date"] = forecast_dates
                
                # Combine historical and forecast
                combined_df = combine_historical_and_forecast(data, best_forecast, start_date, data.index[-1])
                fig_combo = px.line(combined_df, x="Date", y="Price", color="Type", title=f"{ticker}: Historical & 1-Year Forecast")
                st.plotly_chart(fig_combo, use_container_width=True)
        with col2:
            st.markdown("### Explore More")
            # Dummy related companies info
            st.write("**Microsoft Corp (MSFT)** +0.78%")
            st.write("**Amazon Inc (AMZN)** +0.88%")
            st.write("**Alphabet Inc (GOOGL)** +1.25%")
            st.write("**NVIDIA Corp (NVDA)** +1.66%")
        
        st.markdown("---")
        
        # Company Info based on ticker (dynamic)
        info = COMPANY_INFO.get(ticker, {
            "Name": ticker,
            "CEO": "N/A",
            "Founded": "N/A",
            "Headquarters": "N/A",
            "Revenue": "N/A",
            "Employees": "N/A",
            "Description": "Company information not available."
        })
        st.subheader(f"{info['Name']} Company Info")
        st.write(info["Description"])
        col_info1, col_info2 = st.columns(2)
        with col_info1:
            st.write(f"**CEO:** {info['CEO']}")
            st.write(f"**Founded:** {info['Founded']}")
            st.write(f"**Headquarters:** {info['Headquarters']}")
        with col_info2:
            st.write(f"**Revenue:** {info['Revenue']}")
            st.write(f"**Employees:** {info['Employees']}")
        
        st.markdown("---")
        st.subheader("Latest News")
        news_df = get_news_summaries(news_items)
        if not news_df.empty:
            for idx, row in news_df.iterrows():
                with st.expander(row['Title']):
                    st.write(row['Summary'])
        else:
            st.write("No news items available.")
    
    # ---------------------
    # TAB: Charts & Indicators
    # ---------------------
    with tabs[1]:
        st.header("Historical Performance & Technical Indicators")
        price_min = float(data['Close'].min())
        price_max = float(data['Close'].max())
        selected_range = st.slider("Select Closing Price Range", min_value=price_min, max_value=price_max, value=(price_min, price_max))
        filtered_data = data[(data['Close'] >= selected_range[0]) & (data['Close'] <= selected_range[1])]
        chart_data = filtered_data.reset_index()[['Date', 'Close']].dropna()
        if chart_data.empty:
            st.error("No chart data available for the selected range.")
        else:
            try:
                fig_line = px.line(chart_data, x="Date", y="Close", title="Historical Closing Prices", labels={"Close": "Price"})
                st.plotly_chart(fig_line, use_container_width=True)
            except Exception as e:
                st.error(f"Error rendering chart: {e}")
        features = additional_interactive_features(data.copy())
        st.subheader("Additional Indicators")
        if show_ma:
            st.plotly_chart(features['ma_chart'], use_container_width=True)
        if show_volatility:
            st.plotly_chart(features['vol_chart'], use_container_width=True)
        if 'rsi_chart' in features:
            st.plotly_chart(features['rsi_chart'], use_container_width=True)
        if 'macd_chart' in features:
            st.plotly_chart(features['macd_chart'], use_container_width=True)
        if 'macd_hist_chart' in features:
            st.plotly_chart(features['macd_hist_chart'], use_container_width=True)
        st.subheader("Recent 30-Day Data")
        st.dataframe(features['recent_table'])
    
    # ---------------------
    # TAB: Forecast
    # ---------------------
    with tabs[2]:
        st.header("Stock Price Forecast")
        st.write("Forecasting using Prophet, ARIMA, and LSTM models with hyper-parameter tuning.")
        prophet_params = tune_prophet(data)
        arima_params = tune_arima(data['Close'])
        lstm_params = tune_lstm(data['Close'])
        try:
            prophet_result = forecast_prophet(data, forecast_days, tuned_params=prophet_params)
        except Exception as e:
            st.error(f"Prophet forecasting failed: {e}")
            prophet_result = pd.DataFrame({"forecast": np.zeros(forecast_days)})
        try:
            arima_result = forecast_arima(data['Close'], forecast_days, tuned_params=arima_params)
        except Exception as e:
            st.error(f"ARIMA forecasting failed: {e}")
            arima_result = pd.DataFrame({"forecast": np.zeros(forecast_days)})
        try:
            lstm_result = forecast_lstm(data['Close'], forecast_days, tuned_params=lstm_params)
        except Exception as e:
            st.error(f"LSTM forecasting failed: {e}")
            lstm_result = pd.DataFrame({"forecast": np.zeros(forecast_days)})
        
        # For demonstration, select Prophet as best model (could be chosen based on error metrics)
        best_result = prophet_result.copy()
        best_result["forecast"] = best_result["forecast"].round(2) * sentiment_factor
        forecast_dates = pd.date_range(start=end_date, periods=forecast_days+1)[1:]
        best_result["Date"] = forecast_dates
        st.success(f"Forecast generated (Model: Prophet)")
        st.dataframe(best_result.round(2).style.format({"forecast": "${:,.2f}"}))
        
        # Forecast Chart
        forecast_chart_data = best_result.melt(id_vars="Date", value_vars=["forecast"], var_name="Type", value_name="Price")
        try:
            fig_forecast = px.line(forecast_chart_data, x="Date", y="Price", color="Type", title=f"{ticker} {forecast_days}-Day Forecast")
            st.plotly_chart(fig_forecast, use_container_width=True)
        except Exception as e:
            st.error(f"Error rendering forecast chart: {e}")
    
    # ---------------------
    # TAB: News Impact
    # ---------------------
    with tabs[3]:
        st.header("News Impact")
        news_df = get_news_summaries(news_items)
        if not news_df.empty:
            for idx, row in news_df.iterrows():
                with st.expander(row['Title']):
                    st.write(row['Summary'])
        else:
            st.write("No news items available.")
    
    # ---------------------
    # TAB: Insights
    # ---------------------
    with tabs[4]:
        st.header("Insights & Recommendations")
        st.markdown("""
        **Market Analysis:**
        - Positive sentiment indicates potential upward momentum.
        - Negative sentiment signals caution.
        - Technical indicators (RSI, MACD) enhance understanding of market trends.
        - External factors like economic and political news also influence the market.
        
        **Recommendations:**
        - Consider buying if sentiment is positive and technicals are strong.
        - Exercise caution or consider selling if sentiment is negative or indicators are overbought.
        """)
        st.markdown("### Ask a Question")
        question = st.text_input("Enter your question about market trends or performance:")
        if st.button("Get Answer"):
            if "increase" in question.lower():
                st.write("Stocks may increase if positive sentiment, robust earnings, and supportive technical indicators persist.")
            elif "decrease" in question.lower():
                st.write("Stocks might decrease if negative news and bearish technical signals continue.")
            else:
                st.write("Please provide more details or ask another question.")
    
    # ---------------------
    # TAB: Detailed Analysis
    # ---------------------
    with tabs[5]:
        st.header("Detailed Data Analysis")
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
            st.subheader("Price Distribution")
            try:
                fig_hist = px.histogram(detailed_data.reset_index(), x="Close", nbins=30, title="Closing Price Distribution")
                st.plotly_chart(fig_hist, use_container_width=True)
            except Exception as e:
                st.error(f"Error rendering histogram: {e}")
    
    # ---------------------
    # TAB: Settings
    # ---------------------
    with tabs[6]:
        st.header("Application Settings")
        if st.checkbox("Show raw data"):
            st.subheader("Raw Data")
            st.dataframe(data.round(2))
        st.markdown("### Model Settings")
        st.markdown("Forecasting model parameters can be adjusted in future updates.")

if __name__ == "__main__":
    main()
