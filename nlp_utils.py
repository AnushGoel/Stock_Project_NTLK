import nltk
import logging
import pandas as pd
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Ensure required NLTK resources are available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

def fetch_news(ticker):
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
    analyzer = SentimentIntensityAnalyzer()
    scores = [analyzer.polarity_scores(article.get("content", ""))["compound"] for article in news_items]
    avg_score = np.mean(scores) if scores else 0.0
    logging.info("Calculated average sentiment score: %f", avg_score)
    return avg_score

def summarize_text(text, sentences_count=2):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = TextRankSummarizer()
    summary = summarizer(parser.document, sentences_count)
    summary_text = " ".join(str(sentence) for sentence in summary)
    logging.info("Summarized text: %s", summary_text)
    return summary_text

def get_news_summaries(news_items):
    summaries = []
    for article in news_items:
        title = article.get("title", "No Title")
        content = article.get("content", "")
        summary = summarize_text(content, sentences_count=2)
        summaries.append({"Title": title, "Summary": summary})
    df = pd.DataFrame(summaries)
    logging.info("Generated news summaries DataFrame with %d records", len(df))
    return df
