
# 📊 StockGPT: AI-Powered Financial Forecasting and Sentiment Analysis Framework

<p align="center">
  <a href="https://stockpy-8gheu62lbnkprwkuw75rvb.streamlit.app/" target="_blank">
    <img src="https://img.shields.io/badge/Launch-App-brightgreen?style=for-the-badge&logo=streamlit" alt="Launch App">
  </a>
</p>


**StockGPT** is a research-grade platform for integrated **stock market forecasting**, **technical analysis**, and **news sentiment extraction** using **machine learning**, **natural language processing (NLP)**, and **financial time series modeling**.

Developed as a modular and extensible framework, StockGPT serves as a robust experimentation and evaluation environment for financial data scientists, quants, and researchers seeking to bridge the gap between market data and intelligent investment strategies.

---

## 🧠 Core Capabilities

### 🔍 Data Ingestion & Preprocessing
- Retrieves historical stock data via [Yahoo Finance API](https://pypi.org/project/yfinance/)
- Flattens and cleanses data for modeling
- Computes technical indicators:
  - **RSI (Relative Strength Index)**
  - **MACD (Moving Average Convergence Divergence)**
  - **Moving Averages**
  - **Rolling Volatility**

### 📈 Forecasting Engine
Three forecast models are implemented with tunable parameters:
- **Facebook Prophet**
- **ARIMA (AutoRegressive Integrated Moving Average)**
- **LSTM Neural Networks** via TensorFlow/Keras

Each model is independently tunable and benchmarked using **Mean Absolute Error (MAE)** for automatic model selection.

### 📰 NLP & Sentiment Analysis
- Real-time financial news fetched via **NewsAPI**
- Summarized using **TextRank** algorithm (Sumy)
- Sentiment scored using **VADER (Valence Aware Dictionary & sEntiment Reasoner)** from NLTK
- Aggregated sentiment score is factored into forecast adjustments (+/- 5% correction)

### 📊 Visualization and UI
- **Streamlit** front-end
- Modular tabs:
  - Company Overview
  - Historical & Technical Charting
  - Multi-Model Forecast Visualization
  - News Sentiment Breakdown
  - Comparative Analysis (side-by-side)
  - Insights & Recommendations
  - Raw Data & Custom Settings

---

## 📁 Project Structure

```
StockGPT/
├── stock.py                # Main Streamlit application
├── forecast_models.py      # Prophet, ARIMA, and LSTM models
├── model_tuning.py         # Model-specific hyperparameter tuning logic
├── additional_factors.py   # RSI, MACD, and other technical indicators
├── nlp_utils.py            # News fetching, summarization, sentiment scoring
├── requirements.txt        # Python dependency list
```

---

## ⚙️ Installation

```bash
git clone https://github.com/your-org/StockGPT.git
cd StockGPT
pip install -r requirements.txt
```

Ensure you have:
- Python 3.7+
- Valid API Key for [NewsAPI](https://newsapi.org/) set in `nlp_utils.py` (`API_KEY` variable)

---

## 🚀 Running the App

```bash
streamlit run stock.py
```

---

## 🧪 Research Value & Applications

This project supports:
- Time-series model benchmarking
- Sentiment-aware price movement forecasting
- Financial news signal extraction
- Explainable AI-driven investment research
- Interactive exploration of alternative forecast models

Ideal for academic labs, research institutions, fintech product R&D, and investor tooling.

---

## 🧰 Key Dependencies

- `streamlit` - UI Framework
- `yfinance` - Stock data provider
- `prophet`, `statsmodels`, `tensorflow` - Forecasting models
- `nltk`, `sumy` - NLP
- `plotly` - Interactive charting

See `requirements.txt` for full list.

---


---

## 🚀 Live Project

Click below to launch the deployed Streamlit app:


<p align="center">
  <a href="https://stockpy-8gheu62lbnkprwkuw75rvb.streamlit.app/" target="_blank">
    <img src="https://img.shields.io/badge/Launch-App-brightgreen?style=for-the-badge&logo=streamlit" alt="Launch App">
  </a>
</p>


---

## 📜 License

Released under the **MIT License**. For research and educational use.

---

## 📚 Citation

If you use **StockGPT** in your research, please consider referencing the following works for related context and methodological inspiration:

> **Koa, K.J., Ma, Y., Ng, R., & Chua, T.S.** (2024, May). Learning to generate explainable stock predictions using self-reflective large language models. *Proceedings of the ACM Web Conference 2024*, 4304–4315. [https://dl.acm.org/doi/pdf/10.1145/3589334.3645611](https://dl.acm.org/doi/pdf/10.1145/3589334.3645611)

> **Thomas, P., Spielman, S., Craswell, N., & Mitra, B.** (2024, July). Large language models can accurately predict searcher preferences. *Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval*, 1930–1940. [https://dl.acm.org/doi/pdf/10.1145/3626772.3657707](https://dl.acm.org/doi/pdf/10.1145/3626772.3657707)


---

## ✉️ Contact

For collaborations, questions, or contributions, feel free to open an issue or contact the maintainers.
