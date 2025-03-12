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

def summarize_text(text, sentences_count=2):
    """
    Summarize text using Sumy's TextRank algorithm.
    If a LookupError occurs, attempt to download the resource and retry.
    On failure, fall back to a basic summarization by splitting on periods.
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
    Ensures each news article is a dictionary before processing.
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
    Only processes items that are dictionaries.
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

def fetch_news(ticker):
    """
    Return a list of dummy news items covering a wide range of topics.
    Topics include company news, AI, economic, sociopolitical, geopolitical, expansion, market situation, and employment.
    """
    try:
        dummy_news = [
            {"title": f"{ticker} Hits Record High", 
             "content": f"{ticker} has reached a record high due to strong earnings and optimistic investor sentiment."},
            {"title": f"Concerns Over {ticker}'s Supply Chain", 
             "content": f"Disruptions in the supply chain for {ticker} could impact its future performance."},
            {"title": f"{ticker} Announces New Product Line", 
             "content": f"{ticker} is launching a new product line expected to boost its market share."},
            {"title": "Economic Growth Slows Down", 
             "content": "Recent economic reports indicate a slowdown in growth, which may affect consumer spending."},
            {"title": "Political Uncertainty in Key Markets", 
             "content": "Political instability in major markets is raising concerns about regulatory changes."},
            {"title": "Global Trade Tensions Ease", 
             "content": "Improvements in global trade negotiations could benefit multinational companies."},
            {"title": "Tech Sector Faces Regulatory Scrutiny", 
             "content": "New regulations are being considered that could impact tech companies' operations."},
            {"title": "AI Breakthroughs Drive Innovation", 
             "content": "Recent advancements in AI technology are reshaping the competitive landscape."},
            {"title": "Company Expansion Plans Announced", 
             "content": f"{ticker} revealed plans to expand into new international markets."},
            {"title": "Market Volatility Increases Amid Uncertainty", 
             "content": "Heightened volatility in the market is causing investors to reassess risk."},
            {"title": "Employment Figures Exceed Expectations", 
             "content": "Strong employment data is boosting consumer confidence and market sentiment."},
            {"title": "Social Trends Impacting Consumer Behavior", 
             "content": "Changes in social behavior are influencing purchasing patterns across various sectors."},
            {"title": "Geopolitical Tensions Rise", 
             "content": "Increasing geopolitical tensions are adding uncertainty to the global economy."},
            {"title": "Expansion Plans Drive Future Growth", 
             "content": f"{ticker} is planning significant expansion, potentially leading to increased revenues."},
            {"title": "Market Situation Update: Mixed Signals", 
             "content": "The current market shows mixed signals, suggesting cautious optimism among investors."},
            {"title": "Employment Growth Signals Economic Resilience", 
             "content": "Continued growth in employment numbers points to a resilient economy."},
            {"title": "Tech Innovation Fuels Industry Transformation", 
             "content": "Innovative technologies are disrupting traditional business models in the tech sector."},
            {"title": "Investor Caution Amid Global Uncertainty", 
             "content": "Global economic and political uncertainties are prompting investors to adopt a cautious stance."},
            {"title": "Breakthrough in AI Research", 
             "content": "A major breakthrough in AI could have far-reaching implications for the industry."},
            {"title": "New Market Entry Strategies", 
             "content": f"{ticker} is exploring new market entry strategies to capture emerging opportunities."},
            # You can add additional items here to reach 50-60 news items if desired.
        ]
        logging.info("Fetched extended dummy news for ticker: %s", ticker)
        return dummy_news
    except Exception as e:
        logging.error("Error in fetch_news: %s", e)
        return []
