# nlp_utils.py
import nltk
import logging
import pandas as pd
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer

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

def summarize_text(text, sentences_count=2):
    """
    Summarize text using Sumy's TextRank algorithm.
    If a LookupError occurs, attempt to download the resource and retry.
    On failure, fall back to a basic summarization.
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
    summaries = []
    for article in news_items:
        title = article.get("title", "No Title")
        content = article.get("content", "")
        summary = summarize_text(content, sentences_count=2)
        summaries.append({"Title": title, "Summary": summary})
    df = pd.DataFrame(summaries)
    logging.info("Generated news summaries DataFrame with %d records", len(df))
    return df

def sentiment_analysis(news_items):
    try:
        analyzer = SentimentIntensityAnalyzer()
        scores = [analyzer.polarity_scores(article.get("content", ""))["compound"] for article in news_items]
        avg_score = np.mean(scores) if scores else 0.0
        logging.info("Calculated average sentiment score: %f", avg_score)
        return avg_score
    except Exception as e:
        logging.error("Error in sentiment_analysis: %s", e)
        return 0.0

def fetch_news(ticker):
    """
    Return a list of dummy news items including economic and political news.
    """
    try:
        dummy_news = [
            {"title": f"{ticker} hits record high in US markets", 
             "content": f"{ticker} has reached a record high due to strong earnings and positive investor sentiment."},
            {"title": f"Concerns over {ticker}'s supply chain", 
             "content": f"Reports indicate disruptions in the supply chain for {ticker}, possibly affecting future performance."},
            {"title": f"{ticker} announces new product line", 
             "content": f"{ticker} is launching a new product line expected to boost sales and market share."},
            {"title": "Economic Growth Slows", 
             "content": "Recent economic reports indicate slower growth, which could affect consumer spending and overall market conditions."},
            {"title": "Political Uncertainty in Key Markets", 
             "content": "Political instability in major markets is raising concerns among investors about potential regulatory changes."},
            {"title": "Global Trade Tensions Ease", 
             "content": "Improvement in global trade negotiations is expected to benefit multinational companies and overall market sentiment."},
        ]
        logging.info("Fetched extended dummy news for ticker: %s", ticker)
        return dummy_news
    except Exception as e:
        logging.error("Error in fetch_news: %s", e)
        return []
