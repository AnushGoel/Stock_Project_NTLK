import requests
import nltk
import logging
import pandas as pd
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Ensure required NLTK resources are available.
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

# Your NewsAPI key (store securely in production)
API_KEY = "a9d34b39961346bca7b51ea428732d84"

def fetch_news(ticker):
    """
    Fetch news articles from NewsAPI for the given ticker.
    Returns a list of dictionaries with keys 'title' and 'content'.
    """
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": ticker,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 50,
        "apiKey": API_KEY
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        articles = data.get("articles", [])
        result = []
        for article in articles:
            result.append({
                "title": article.get("title", "No Title"),
                "content": article.get("description", "") or article.get("content", "")
            })
        logging.info("Fetched %d news articles for ticker: %s", len(result), ticker)
        return result
    except Exception as e:
        logging.error(f"Error fetching news from NewsAPI: {e}")
        return []

def summarize_text(text, sentences_count=2):
    """
    Summarize text using Sumy's TextRank algorithm.
    If a LookupError occurs, attempt to download required resources and retry.
    Falls back to a basic sentence-split summarization if needed.
    """
    try:
        from sumy.parsers.plaintext import PlaintextParser
        from sumy.nlp.tokenizers import Tokenizer
        from sumy.summarizers.text_rank import TextRankSummarizer
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = TextRankSummarizer()
        summary = summarizer(parser.document, sentences_count)
        summary_text = " ".join(str(sentence) for sentence in summary)
        return summary_text
    except LookupError as e:
        logging.warning("LookupError in summarize_text: %s", e)
        try:
            nltk.download('punkt')
            from sumy.parsers.plaintext import PlaintextParser
            from sumy.nlp.tokenizers import Tokenizer
            from sumy.summarizers.text_rank import TextRankSummarizer
            parser = PlaintextParser.from_string(text, Tokenizer("english"))
            summarizer = TextRankSummarizer()
            summary = summarizer(parser.document, sentences_count)
            summary_text = " ".join(str(sentence) for sentence in summary)
            return summary_text
        except Exception as ex:
            logging.error("Fallback summarization triggered: %s", ex)
            sentences = text.split('. ')
            fallback_summary = '. '.join(sentences[:sentences_count])
            if not fallback_summary.endswith('.'):
                fallback_summary += '.'
            return fallback_summary
    except Exception as e:
        logging.error("Error in summarize_text: %s", e)
        return text

def get_news_summaries(news_items):
    """
    Create a DataFrame containing news titles and summaries.
    Ensures that each news article is a dictionary.
    """
    summaries = []
    for article in news_items:
        if not isinstance(article, dict):
            try:
                article = dict(article)
            except Exception as e:
                logging.error(f"Error converting article to dict: {e}")
                continue
        title = article.get("title", "No Title")
        content = article.get("content", "")
        summary = summarize_text(content, sentences_count=2)
        summaries.append({"Title": title, "Summary": summary})
    df = pd.DataFrame(summaries)
    logging.info("Generated news summaries DataFrame with %d records", len(df))
    return df

def sentiment_analysis(news_items):
    """
    Compute the average sentiment score from the provided news items.
    Processes only items that are dictionaries.
    """
    try:
        analyzer = SentimentIntensityAnalyzer()
        scores = [
            analyzer.polarity_scores(article.get("content", ""))["compound"]
            for article in news_items if isinstance(article, dict)
        ]
        avg_score = np.mean(scores) if scores else 0.0
        logging.info("Calculated average sentiment score: %f", avg_score)
        return avg_score
    except Exception as e:
        logging.error("Error in sentiment_analysis: %s", e)
        return 0.0
