from transformers import pipeline
import pandas as pd
import logging
from typing import Tuple
from tqdm import tqdm

class SentimentAnalyzer:
    def __init__(self, model_name="distilbert-base-uncased-finetuned-sst-2-english"):
        self.logger = logging.getLogger(__name__)
        self.model = pipeline("sentiment-analysis", model=model_name)

    def analyze_review(self, text: str) -> Tuple[str, float]:
        """Analyze single review"""
        try:
            result = self.model(text)[0]
            return result['label'], result['score']
        except Exception as e:
            self.logger.error(f"Sentiment analysis failed: {str(e)}")
            return "ERROR", 0.0

    def analyze_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze all reviews in DataFrame"""
        tqdm.pandas(desc="Analyzing sentiment")
        df[['sentiment', 'sentiment_score']] = df['review'].progress_apply(
            lambda x: pd.Series(self.analyze_review(x))
        )
        return df