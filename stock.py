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

def flatten_data_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Flatten multi-index columns if present."""
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data

def get_company_info(ticker: str) -> dict:
    """Fetch company info from yfinance Ticker object."""
    try:
        tk = yf.Ticker(ticker)
        info = tk.info
        # Some info might be missing, so provide fallbacks.
        return {
            "longBusinessSummary": info.get("longBusinessSummary", "No description available."),
            "website": info.get("website", "N/A"),
            "logo_url": info.get("logo_url", "https://via.placeholder.com/150")
        }
    except Exception as e:
        logging.error(f"Error fetching company info for {ticker}: {e}")
        return {"longBusinessSummary": "No description available.", "website": "N/A", "logo_url": "https://via.placeholder.com/150"}

def additional_interactive_features(data: pd.DataFrame) -> dict:
    """Generate additional charts and recent data table."""
    features = {}
    data_calc = data.copy()
    features['recent_table'] = data_calc.tail(30).round(2)
    
    # Calculate Moving Average
    data_calc['MA20'] = data_calc['Close'].rolling(window=20).mean()
    fig_ma = go.Figure()
    fig_ma.add_trace(go.Scatter(x=data_calc.index, y=data_calc['MA20'].round(2), mode='lines', name='MA20', line=dict(color='red')))
    fig_ma.update_layout(title="20-Day Moving Average", xaxis_title="Date", yaxis_title="MA20")
    features['ma_chart'] = fig_ma
    
    # Calculate Volatility
    data_calc['Volatility'] = data_calc['Close'].rolling(window=20).std()
    fig_vol = go.Figure()
    fig_vol.add_trace(go.Scatter(x=data_calc.index, y=data_calc['Volatility'].round(2), mode='lines', name='Volatility', line=dict(color='orange')))
    fig_vol.update_layout(title="20-Day Volatility", xaxis_title="Date", yaxis_title="Volatility")
    features['vol_chart'] = fig_vol
    
    # RSI, MACD, etc. (if computed)
    if 'RSI' in data_calc.columns:
        fig_rsi = px.line(data_calc.reset_index(), x="Date", y="RSI", title="RSI Over Time")
        features['rsi_chart'] = fig_rsi
    if 'MACD' in data_calc.columns and 'MACD_Signal' in data_calc.columns:
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=data_calc.index, y=data_calc['MACD'].round(2), mode='lines', name='MACD'))
        fig_macd.add_trace(go.Scatter(x=data_calc.index, y=data_calc['MACD_Signal'].round(2), mode='lines', name='Signal'))
        fig_macd.update_layout(title="MACD & Signal", xaxis_title="Date", yaxis_title="MACD")
        features['macd_chart'] = fig_macd
    if 'MACD_Hist' in data_calc.columns:
        fig_hist = px.bar(data_calc.reset_index(), x="Date", y="MACD_Hist", title="MACD Histogram")
        features['macd_hist_chart'] = fig_hist

    return features

def main():
    st.set_page_config(page_title="📈 Advanced StockGPT", layout="wide")
    
    # Sidebar: Input only ticker and date range.
    ticker = st.sidebar.text_input("📌 Stock Ticker:", "AAPL").upper()
    start_date = st.sidebar.date_input("📅 Start Date", datetime.date.today() - datetime.timedelta(days=365))
    end_date = datetime.date.today()
    forecast_days = st.sidebar.slider("Forecast Days", 7, 60, 30)
    
    # Tabs for different sections.
    tabs = st.tabs(["📊 Dashboard", "📈 Charts", "🚀 Forecast", "📰 News Impact", "💡 Insights", "📌 Detailed Analysis", "⚙️ Settings"])
    
    # Fetch stock data.
    data_load_state = st.info("Fetching stock data...")
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            st.error("No data found for ticker. Check the symbol and try again.")
            return
        data = flatten_data_columns(data)
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return
    data_load_state.success("Data fetched successfully!")
    data.index.name = "Date"
    
    # Compute technical indicators.
    data = calculate_technical_indicators(data)
    
    # Fetch company info.
    comp_info = get_company_info(ticker)
    
    # Fetch news and sentiment.
    news_items = fetch_news(ticker)
    sentiment_score = sentiment_analysis(news_items)
    sentiment_factor = 1 + (sentiment_score * 0.05)
    
    # ===========================
    # TAB: Dashboard (Main Page)
    # ===========================
    with tabs[0]:
        st.markdown("<style> .main-header {font-size: 2.5rem; font-weight: bold; color: #333;} </style>", unsafe_allow_html=True)
        st.markdown(f"<div class='main-header'>{ticker} Company Overview</div>", unsafe_allow_html=True)
        
        # Show company logo and info in two columns.
        col1, col2 = st.columns([1, 3])
        with col1:
            st.image(comp_info["logo_url"], width=100)
        with col2:
            st.markdown(f"**Website:** [Visit]({comp_info['website']})" if comp_info["website"] != "N/A" else "Website: N/A")
            st.markdown(comp_info["longBusinessSummary"])
        
        st.markdown("---")
        
        # Instead of a separate close-price graph, display an interactive candlestick chart.
        try:
            candlestick_data = data.reset_index()
            fig_candle = go.Figure(data=[go.Candlestick(
                x=candlestick_data["Date"],
                open=candlestick_data["Open"].round(2),
                high=candlestick_data["High"].round(2),
                low=candlestick_data["Low"].round(2),
                close=candlestick_data["Close"].round(2),
                increasing_line_color="green", decreasing_line_color="red",
                hovertemplate=
                    "Date: %{x}<br>" +
                    "Open: $%{open:.2f}<br>" +
                    "High: $%{high:.2f}<br>" +
                    "Low: $%{low:.2f}<br>" +
                    "Close: $%{close:.2f}<extra></extra>"
            )])
            fig_candle.update_layout(title=f"{ticker} Candlestick Chart", xaxis_title="Date", yaxis_title="Price")
            st.plotly_chart(fig_candle, use_container_width=True)
        except Exception as e:
            st.error(f"Error rendering candlestick chart: {e}")
        
        # Display a combined historical chart with forecast.
        st.markdown("### Historical Data & Forecast")
        # Basic tuning and forecast for a short period.
        prophet_params = tune_prophet(data)
        try:
            forecast_result = forecast_prophet(data, forecast_days, tuned_params=prophet_params)
        except Exception as e:
            st.error(f"Forecasting failed: {e}")
            forecast_result = pd.DataFrame({"forecast": np.zeros(forecast_days), "lower": np.zeros(forecast_days), "upper": np.zeros(forecast_days)})
        # Apply sentiment factor and round numbers.
        forecast_result["forecast"] = (forecast_result["forecast"] * sentiment_factor).round(2)
        forecast_result["lower"] = (forecast_result["lower"] * sentiment_factor).round(2)
        forecast_result["upper"] = (forecast_result["upper"] * sentiment_factor).round(2)
        forecast_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=forecast_days)
        forecast_result["Date"] = forecast_dates
        
        # Prepare historical data.
        hist_data = data.reset_index()[["Date", "Close"]].copy()
        hist_data["Close"] = hist_data["Close"].round(2)
        hist_data["Type"] = "Historical"
        hist_data.rename(columns={"Close": "Price"}, inplace=True)
        fc_data = forecast_result[["Date", "forecast"]].copy()
        fc_data["Type"] = "Forecast"
        fc_data.rename(columns={"forecast": "Price"}, inplace=True)
        combined = pd.concat([hist_data, fc_data], ignore_index=True)
        
        fig_combined = px.line(combined, x="Date", y="Price", color="Type",
                               title=f"{ticker} Historical vs. {forecast_days}-Day Forecast")
        st.plotly_chart(fig_combined, use_container_width=True)
        
    # ===========================
    # TAB: Charts (Indicators)
    # ===========================
    with tabs[1]:
        st.header("Historical Stock Performance & Indicators")
        price_min = float(data['Close'].min())
        price_max = float(data['Close'].max())
        selected_range = st.slider("Select Price Range", min_value=price_min, max_value=price_max, value=(price_min, price_max))
        filtered_data = data[(data['Close'] >= selected_range[0]) & (data['Close'] <= selected_range[1])]
        chart_data = filtered_data.reset_index()[["Date", "Close"]].dropna()
        if chart_data.empty:
            st.error("No chart data available for the selected range.")
        else:
            try:
                fig_line = px.line(chart_data, x="Date", y="Close", title="Historical Closing Prices", labels={"Close": "Price ($)"})
                st.plotly_chart(fig_line, use_container_width=True)
            except Exception as e:
                st.error(f"Error rendering chart: {e}")
        features = additional_interactive_features(data.copy())
        st.subheader("Recent Prices (Last 30 Days)")
        st.dataframe(features['recent_table'])
        if "ma_chart" in features:
            st.subheader("20-Day Moving Average")
            st.plotly_chart(features['ma_chart'], use_container_width=True)
        if "vol_chart" in features:
            st.subheader("20-Day Volatility")
            st.plotly_chart(features['vol_chart'], use_container_width=True)
        if "rsi_chart" in features:
            st.subheader("RSI Chart")
            st.plotly_chart(features['rsi_chart'], use_container_width=True)
        if "macd_chart" in features:
            st.subheader("MACD & Signal")
            st.plotly_chart(features['macd_chart'], use_container_width=True)
        if "macd_hist_chart" in features:
            st.subheader("MACD Histogram")
            st.plotly_chart(features['macd_hist_chart'], use_container_width=True)
    
    # ===========================
    # TAB: Forecast (Detailed)
    # ===========================
    with tabs[2]:
        st.header("Forecast Details")
        st.write("Forecasting using Prophet, ARIMA, and LSTM models.")
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
        
        # For demonstration, choose best model based on simulated MAE.
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
        best_result = {"Prophet": prophet_result, "ARIMA": arima_result, "LSTM": lstm_result}[best_model]
        best_result_adj = best_result.copy()
        best_result_adj["forecast"] = (best_result_adj["forecast"] * sentiment_factor).round(2)
        best_result_adj["lower"] = (best_result_adj["lower"] * sentiment_factor).round(2)
        best_result_adj["upper"] = (best_result_adj["upper"] * sentiment_factor).round(2)
        forecast_dates = pd.date_range(start=end_date, periods=forecast_days+1)[1:]
        best_result_adj["Date"] = forecast_dates
        st.success(f"Best Forecast Model: **{best_model}** | MAE: {errors[best_model]:.2f}")
        st.dataframe(best_result_adj.round(2).style.format({"forecast": "${:,.2f}", "lower": "${:,.2f}", "upper": "${:,.2f}"}))
        
        fc_chart_data = best_result_adj.melt(id_vars="Date", value_vars=["forecast", "lower", "upper"], var_name="Type", value_name="Price")
        try:
            fig_fc = px.line(fc_chart_data, x="Date", y="Price", color="Type", title=f"{ticker} Forecast ({forecast_days}-Day)")
            st.plotly_chart(fig_fc, use_container_width=True)
        except Exception as e:
            st.error(f"Error rendering forecast chart: {e}")
    
    # ===========================
    # TAB: News Impact
    # ===========================
    with tabs[3]:
        st.header("News Impact")
        news_df = get_news_summaries(news_items)
        if not news_df.empty:
            for idx, row in news_df.iterrows():
                with st.expander(row['Title']):
                    # Display a small placeholder image along with news (you can replace with actual URLs)
                    st.image("https://via.placeholder.com/100", width=100)
                    st.write(row['Summary'])
        else:
            st.write("No news items available.")
    
    # ===========================
    # TAB: Insights
    # ===========================
    with tabs[4]:
        st.header("Insights & Recommendations")
        st.markdown("""
        **Market Analysis:**
        - Positive sentiment may indicate upward momentum.
        - Negative sentiment is a warning signal.
        - Technical indicators like RSI and MACD offer additional context.
        - Broader economic and political news can also impact markets.
        
        **Recommendations:**
        - Consider buying if sentiment is positive and fundamentals are strong.
        - Exercise caution or consider selling if sentiment is negative.
        """)
        st.markdown("### Ask a Question")
        question = st.text_input("Your question about market trends or stock performance:")
        if st.button("Get Answer"):
            if "increase" in question.lower():
                st.write("Based on current indicators, stocks may increase if positive sentiment and strong earnings persist.")
            elif "decrease" in question.lower():
                st.write("Stocks might decrease if negative news and bearish signals continue.")
            else:
                st.write("Please provide more details for a specific analysis.")
    
    # ===========================
    # TAB: Detailed Analysis
    # ===========================
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
            st.subheader("Distribution of Closing Prices")
            try:
                fig_dist = px.histogram(detailed_data.reset_index(), x="Close", nbins=30, title="Price Distribution")
                st.plotly_chart(fig_dist, use_container_width=True)
            except Exception as e:
                st.error(f"Error rendering histogram: {e}")
    
    # ===========================
    # TAB: Settings
    # ===========================
    with tabs[6]:
        st.header("Settings")
        st.markdown("View raw data and adjust parameters.")
        if st.checkbox("Show raw data"):
            st.dataframe(data.round(2))
        st.markdown("### Model Settings")
        st.markdown("Future adjustments for forecasting model parameters can be made here.")

if __name__ == "__main__":
    main()
