import requests
import random
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

# Your NewsAPI key
API_KEY = "a9d34b39961346bca7b51ea428732d84"

def fetch_news(ticker):
    """
    Fetch news articles from NewsAPI using two queries:
    - One for ticker-specific news.
    - One for broader economic, political, geopolitical, market, employment, technology, and AI news.
    Returns up to 150 unique articles.
    """
    base_url = "https://newsapi.org/v2/everything"
    queries = [
        f'"{ticker}"',  # Exact ticker (in quotes for exact match)
        "economy OR political OR geopolitical OR market OR employment OR technology OR AI OR innovation"
    ]
    
    all_articles = []
    
    for q in queries:
        params = {
            "q": q,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": 100,  # Maximum page size
            "apiKey": API_KEY
        }
        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()
            articles = data.get("articles", [])
            all_articles.extend(articles)
        except Exception as e:
            logging.error(f"Error fetching news for query '{q}': {e}")
    
    # Remove duplicates based on title
    seen_titles = set()
    unique_articles = []
    for article in all_articles:
        title = article.get("title", "").strip()
        if title and title not in seen_titles:
            seen_titles.add(title)
            unique_articles.append(article)
    
    # If more than 100 unique articles, randomly sample 100.
    if len(unique_articles) > 100:
        unique_articles = random.sample(unique_articles, 100)
    
    logging.info("Fetched %d unique news articles for ticker: %s", len(unique_articles), ticker)
    return unique_articles

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
    Ensures each news article is a dictionary.
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
