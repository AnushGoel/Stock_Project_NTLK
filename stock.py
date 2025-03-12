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

def get_company_info(ticker: str) -> dict:
    """Fetch company information from yfinance."""
    try:
        tk = yf.Ticker(ticker)
        info = tk.info
        return {
            "longBusinessSummary": info.get("longBusinessSummary", "No description available."),
            "website": info.get("website", "N/A"),
            "logo_url": info.get("logo_url", "https://via.placeholder.com/150"),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "fullTimeEmployees": info.get("fullTimeEmployees", "N/A")
        }
    except Exception as e:
        logging.error(f"Error fetching company info for {ticker}: {e}")
        return {
            "longBusinessSummary": "No description available.",
            "website": "N/A",
            "logo_url": "https://via.placeholder.com/150",
            "sector": "N/A",
            "industry": "N/A",
            "fullTimeEmployees": "N/A"
        }

def additional_interactive_features(data: pd.DataFrame) -> dict:
    """Generate charts (MA, Volatility, RSI, MACD, etc.) and a recent data table."""
    features = {}
    data_calc = data.copy()
    features['recent_table'] = data_calc.tail(30).round(2)
    
    # 20-Day Moving Average
    data_calc['MA20'] = data_calc['Close'].rolling(window=20).mean()
    fig_ma = go.Figure()
    fig_ma.add_trace(go.Scatter(
        x=data_calc.index, 
        y=data_calc['MA20'].round(2), 
        mode='lines', 
        name='MA20', 
        line=dict(color='red')
    ))
    fig_ma.update_layout(title="20-Day Moving Average", xaxis_title="Date", yaxis_title="MA20")
    features['ma_chart'] = fig_ma

    # 20-Day Volatility
    data_calc['Volatility'] = data_calc['Close'].rolling(window=20).std()
    fig_vol = go.Figure()
    fig_vol.add_trace(go.Scatter(
        x=data_calc.index, 
        y=data_calc['Volatility'].round(2), 
        mode='lines', 
        name='Volatility', 
        line=dict(color='orange')
    ))
    fig_vol.update_layout(title="20-Day Volatility", xaxis_title="Date", yaxis_title="Volatility")
    features['vol_chart'] = fig_vol

    # RSI Chart (if available)
    if 'RSI' in data_calc.columns:
        fig_rsi = px.line(data_calc.reset_index(), x="Date", y="RSI", title="RSI Over Time")
        features['rsi_chart'] = fig_rsi

    # MACD & Signal (if available)
    if 'MACD' in data_calc.columns and 'MACD_Signal' in data_calc.columns:
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(
            x=data_calc.index, 
            y=data_calc['MACD'].round(2), 
            mode='lines', 
            name='MACD'
        ))
        fig_macd.add_trace(go.Scatter(
            x=data_calc.index, 
            y=data_calc['MACD_Signal'].round(2), 
            mode='lines', 
            name='Signal'
        ))
        fig_macd.update_layout(title="MACD & Signal", xaxis_title="Date", yaxis_title="MACD")
        features['macd_chart'] = fig_macd

    # MACD Histogram (if available)
    if 'MACD_Hist' in data_calc.columns:
        fig_hist = px.bar(data_calc.reset_index(), x="Date", y="MACD_Hist", title="MACD Histogram")
        features['macd_hist_chart'] = fig_hist

    return features

def combine_historical_and_forecast(data: pd.DataFrame, forecast_df: pd.DataFrame, start_date: datetime.date, end_date: datetime.date) -> pd.DataFrame:
    """
    Merge historical data with forecast data into a single DataFrame for plotting.
    The historical portion is labeled 'Historical' and forecast as 'Forecast'.
    """
    hist_data = data.reset_index()[['Date', 'Close']].copy()
    hist_data = hist_data[(hist_data['Date'] >= pd.to_datetime(start_date)) & (hist_data['Date'] <= pd.to_datetime(end_date))]
    hist_data['Type'] = 'Historical'
    hist_data.rename(columns={'Close': 'Price'}, inplace=True)
    
    fc_data = forecast_df[['Date', 'forecast']].copy()
    fc_data['Type'] = 'Forecast'
    fc_data.rename(columns={'forecast': 'Price'}, inplace=True)

    combined = pd.concat([hist_data, fc_data], ignore_index=True)
    return combined

def main():
    st.set_page_config(page_title="📈 Advanced StockGPT", layout="wide")
    
    # Sidebar: Only Ticker, Date Range, Forecast Days
    ticker = st.sidebar.text_input("📌 Stock Ticker:", "AAPL").upper()
    start_date = st.sidebar.date_input("📅 Start Date", datetime.date.today() - datetime.timedelta(days=365))
    end_date = datetime.date.today()
    forecast_days = st.sidebar.slider("Forecast Days", 7, 60, 30)
    
    # Define tabs (rearranged + new Compare Companies tab)
    tabs = st.tabs([
        "🏢 Company Overview", 
        "📊 Dashboard", 
        "📈 Charts", 
        "🚀 Forecast", 
        "📰 News Impact", 
        "💡 Insights", 
        "📌 Detailed Analysis", 
        "📊 Compare Companies", 
        "⚙️ Settings"
    ])
    
    # Fetch Data
    data_load_state = st.info("Fetching stock data...")
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            st.error("No data found for ticker. Please check the symbol and try again.")
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
    # For demonstration, assume fetch_news now returns a larger list (50-60 items)
    news_items = fetch_news(ticker)
    sentiment_score = sentiment_analysis(news_items)
    sentiment_factor = 1 + (sentiment_score * 0.05)
    
    # ================
    # TAB: Company Overview
    # ================
    with tabs[0]:
        st.markdown("<style>.header {font-size:2.5rem; font-weight:bold; color:#333;}</style>", unsafe_allow_html=True)
        st.markdown("<div class='header'>Company Overview</div>", unsafe_allow_html=True)
        comp_info = get_company_info(ticker)
        st.image(comp_info["logo_url"], width=150)
        if comp_info["website"] != "N/A":
            st.markdown(f"**Website:** [Visit]({comp_info['website']})")
        st.markdown("### Description")
        st.write(comp_info["longBusinessSummary"])
        st.markdown("### Key Information")
        st.write(f"**Sector:** {comp_info.get('sector', 'N/A')}")
        st.write(f"**Industry:** {comp_info.get('industry', 'N/A')}")
        st.write(f"**Employees:** {comp_info.get('fullTimeEmployees', 'N/A')}")
    
    # ================
    # TAB: Dashboard
    # ================
    with tabs[1]:
        st.markdown("<style>.header {font-size:2.5rem; font-weight:bold; color:#333;}</style>", unsafe_allow_html=True)
        st.markdown(f"<div class='header'>{ticker} Dashboard</div>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        with col1:
            try:
                current_price = data['Close'].iloc[-1]
                prev_price = data['Close'].iloc[-2] if len(data) > 1 else current_price
                change = current_price - prev_price
                pct_change = (change / prev_price * 100) if prev_price != 0 else 0
                color_style = "red" if change < 0 else "green"
                
                st.markdown(
                    f"""
                    <div style='background-color:#f9f9f9; border-radius:8px; padding:1rem; margin-bottom:1rem;'>
                        <span style='font-size:2rem; font-weight:bold;'>${current_price:.2f}</span>
                        <span style='font-size:1.2rem; margin-left:1rem; color:{color_style};'>
                            {change:+.2f} ({pct_change:+.2f}%)
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                # Use system's local time (without seconds)
                local_time = datetime.datetime.now().astimezone().strftime("%B %d, %Y %I:%M %p %Z")
                st.caption(f"As of {local_time}")
            except Exception as e:
                st.error(f"Error retrieving price: {e}")
            
            # Display interactive candlestick chart with closing price overlay
            st.markdown("### Interactive Candlestick Chart")
            try:
                candle_data = data.reset_index()
                fig_candle = go.Figure(data=[go.Candlestick(
                    x=candle_data["Date"],
                    open=candle_data["Open"].round(2),
                    high=candle_data["High"].round(2),
                    low=candle_data["Low"].round(2),
                    close=candle_data["Close"].round(2),
                    increasing_line_color="green",
                    decreasing_line_color="red",
                    hoverinfo="text",
                    hovertext=[
                        f"Date: {d.strftime('%Y-%m-%d')}<br>Open: ${o:.2f}<br>High: ${h:.2f}<br>Low: ${l:.2f}<br>Close: ${c:.2f}"
                        for d, o, h, l, c in zip(candle_data["Date"], candle_data["Open"], candle_data["High"], candle_data["Low"], candle_data["Close"])
                    ]
                )])
                # Add overlay line trace for closing price
                fig_candle.add_trace(go.Scatter(
                    x=candle_data["Date"],
                    y=candle_data["Close"].round(2),
                    mode="lines",
                    line=dict(color="blue", dash="dot"),
                    name="Close Price"
                ))
                fig_candle.update_layout(title=f"{ticker} Candlestick Chart", xaxis_title="Date", yaxis_title="Price")
                st.plotly_chart(fig_candle, use_container_width=True)
            except Exception as e:
                st.error(f"Error rendering candlestick chart: {e}")
            
            # Display historical price line chart
            st.markdown("### Historical Price Chart")
            hist_data = data.reset_index()[["Date", "Close"]].copy()
            hist_data["Close"] = hist_data["Close"].round(2)
            fig_hist = px.line(hist_data, x="Date", y="Close", title=f"{ticker} Historical Prices", labels={"Close": "Price ($)"})
            st.plotly_chart(fig_hist, use_container_width=True)
        with col2:
            st.markdown("### Explore More")
            st.write("**Microsoft Corp (MSFT)** +0.78%")
            st.write("**Samsung Electronics** +0.29%")
            st.write("**Amazon Inc (AMZN)** +0.88%")
            st.write("**NVIDIA Corp (NVDA)** +1.66%")
        
        st.markdown("---")
        st.subheader("Latest News")
        news_df = get_news_summaries(news_items)
        if not news_df.empty:
            for idx, row in news_df.iterrows():
                with st.expander(row['Title']):
                    st.write(row['Summary'])
        else:
            st.write("No news items available.")
    
    # ================
    # TAB: Charts (Indicators)
    # ================
    with tabs[2]:
        st.header("Historical Performance & Technical Indicators")
        price_min = float(data["Close"].min())
        price_max = float(data["Close"].max())
        selected_range = st.slider("Select Price Range", min_value=price_min, max_value=price_max, value=(price_min, price_max))
        filtered_data = data[(data["Close"] >= selected_range[0]) & (data["Close"] <= selected_range[1])]
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
        st.dataframe(features["recent_table"])
        if "ma_chart" in features:
            st.subheader("20-Day Moving Average")
            st.plotly_chart(features["ma_chart"], use_container_width=True)
        if "vol_chart" in features:
            st.subheader("20-Day Volatility")
            st.plotly_chart(features["vol_chart"], use_container_width=True)
        if "rsi_chart" in features:
            st.subheader("RSI Chart")
            st.plotly_chart(features["rsi_chart"], use_container_width=True)
        if "macd_chart" in features:
            st.subheader("MACD & Signal")
            st.plotly_chart(features["macd_chart"], use_container_width=True)
        if "macd_hist_chart" in features:
            st.subheader("MACD Histogram")
            st.plotly_chart(features["macd_hist_chart"], use_container_width=True)
    
    # ================
    # TAB: Forecast
    # ================
    with tabs[3]:
        st.header("Forecast Details")
        st.write("Forecasting using Prophet, ARIMA, and LSTM models.")
        prophet_params = tune_prophet(data)
        arima_params = tune_arima(data["Close"])
        lstm_params = tune_lstm(data["Close"])
        try:
            prophet_result = forecast_prophet(data, forecast_days, tuned_params=prophet_params)
        except Exception as e:
            st.error(f"Prophet forecasting failed: {e}")
            prophet_result = pd.DataFrame({"forecast": np.zeros(forecast_days), "lower": np.zeros(forecast_days), "upper": np.zeros(forecast_days)})
        try:
            arima_result = forecast_arima(data["Close"], forecast_days, tuned_params=arima_params)
        except Exception as e:
            st.error(f"ARIMA forecasting failed: {e}")
            arima_result = pd.DataFrame({"forecast": np.zeros(forecast_days), "lower": np.zeros(forecast_days), "upper": np.zeros(forecast_days)})
        try:
            lstm_result = forecast_lstm(data["Close"], forecast_days, tuned_params=lstm_params)
        except Exception as e:
            st.error(f"LSTM forecasting failed: {e}")
            lstm_result = pd.DataFrame({"forecast": np.zeros(forecast_days), "lower": np.zeros(forecast_days), "upper": np.zeros(forecast_days)})
        
        if len(data["Close"]) >= forecast_days:
            actual_recent = data["Close"][-forecast_days:].values
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
        # Apply sentiment factor already computed in best_result_adj if needed
        best_result_adj["forecast"] = best_result_adj["forecast"].round(2)
        best_result_adj["lower"] = best_result_adj["lower"].round(2)
        best_result_adj["upper"] = best_result_adj["upper"].round(2)
        # Create a new Impact Forecast column based on sentiment:
        if sentiment_score >= 0:
            best_result_adj["Impact Forecast"] = (best_result_adj["forecast"] * (1 + abs(sentiment_score) * 0.05)).round(2)
        else:
            best_result_adj["Impact Forecast"] = (best_result_adj["forecast"] * (1 - abs(sentiment_score) * 0.05)).round(2)
        
        forecast_dates = pd.date_range(start=end_date, periods=forecast_days+1)[1:]
        # Rearrange columns: Date, Forecast, Impact Forecast, Lower, Upper
        forecast_df = best_result_adj.copy()
        forecast_df["Date"] = forecast_dates
        forecast_df = forecast_df[["Date", "forecast", "Impact Forecast", "lower", "upper"]]
        st.success(f"Best Forecast Model: **{best_model}** | MAE: {errors[best_model]:.2f} | Sentiment Score: {sentiment_score:.2f}")
        st.dataframe(forecast_df.style.format({
            "forecast": "${:,.2f}", 
            "Impact Forecast": "${:,.2f}",
            "lower": "${:,.2f}", 
            "upper": "${:,.2f}"
        }))
        
        fc_chart_data = forecast_df.melt(id_vars="Date", value_vars=["forecast", "Impact Forecast", "lower", "upper"], var_name="Type", value_name="Price")
        try:
            fig_fc = px.line(fc_chart_data, x="Date", y="Price", color="Type", title=f"{ticker} Forecast Comparison ({forecast_days}-Day)")
            st.plotly_chart(fig_fc, use_container_width=True)
        except Exception as e:
            st.error(f"Error rendering forecast chart: {e}")
    
    # ================
    # TAB: News Impact
    # ================
    with tabs[4]:
        st.header("News Impact")
        # Simulate a larger set of news items (50-60 items)
        news_df = get_news_summaries(news_items)
        st.write(f"Total news items: {len(news_df)}")
        if not news_df.empty:
            for idx, row in news_df.iterrows():
                st.subheader(row['Title'])
                st.write(row['Summary'])
                st.markdown("---")
        else:
            st.write("No news items available.")
    
    # ================
    # TAB: Insights
    # ================
    with tabs[5]:
        st.header("Insights & Recommendations")
        st.markdown("""
        **Market Analysis:**
        - Positive sentiment indicates potential upward momentum.
        - Negative sentiment is a warning signal.
        - Technical indicators (RSI, MACD) provide deeper context.
        - Broader economic and political events also influence market trends.
        
        **Recommendations:**
        - Consider buying if sentiment and indicators are favorable.
        - Exercise caution or consider selling if sentiment is negative.
        """)
        st.markdown("### Ask a Question")
        question = st.text_input("Enter your question about market trends or stock performance:")
        if st.button("Get Answer"):
            if "increase" in question.lower():
                st.write("Stocks may increase if sustained positive sentiment, strong earnings, and favorable technical indicators persist.")
            elif "decrease" in question.lower():
                st.write("Stocks might decrease if negative news and bearish technical indicators continue.")
            else:
                st.write("Please provide more details for a specific analysis.")
    
    # ================
    # TAB: Detailed Analysis
    # ================
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
    
    # ================
    # TAB: Compare Companies
    # ================
    with tabs[7]:
        st.header("Compare Companies")
        col_a, col_b = st.columns(2)
        with col_a:
            ticker1 = st.text_input("Enter first ticker:", value=ticker).upper()
        with col_b:
            ticker2 = st.text_input("Enter second ticker:", value="MSFT").upper()
        
        if ticker1 and ticker2:
            try:
                data1 = yf.download(ticker1, start=start_date, end=end_date)
                data2 = yf.download(ticker2, start=start_date, end=end_date)
                data1 = flatten_data_columns(data1)
                data2 = flatten_data_columns(data2)
                data1.index.name = "Date"
                data2.index.name = "Date"
                df1 = data1.reset_index()[["Date", "Close"]].copy()
                df1["Ticker"] = ticker1
                df2 = data2.reset_index()[["Date", "Close"]].copy()
                df2["Ticker"] = ticker2
                comp_df = pd.concat([df1, df2], ignore_index=True)
                comp_df["Close"] = comp_df["Close"].round(2)
                fig_comp = px.line(comp_df, x="Date", y="Close", color="Ticker", title=f"Comparison: {ticker1} vs. {ticker2}")
                st.plotly_chart(fig_comp, use_container_width=True)
            except Exception as e:
                st.error(f"Error comparing companies: {e}")
    
    # ================
    # TAB: Settings
    # ================
    with tabs[8]:
        st.header("Application Settings")
        st.markdown("View raw data and adjust model parameters (future updates).")
        if st.checkbox("Show raw data"):
            st.dataframe(data.round(2))
        st.markdown("### Model Settings")
        st.markdown("Forecasting model parameters can be adjusted here in future versions.")

if __name__ == "__main__":
    main()
