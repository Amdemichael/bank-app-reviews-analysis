import pandas as pd
from datetime import datetime
from config.settings import PREPROCESSING_CONFIG
import hashlib
import logging
from typing import List, Dict  # Add this import

class ReviewPreprocessor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def _generate_review_id(self, text: str, date: str) -> str:
        """Generate unique ID for each review"""
        return hashlib.md5(f"{text}{date}".encode()).hexdigest()

    def clean_data(self, raw_reviews: List[Dict], bank_name: str) -> pd.DataFrame:
        """Transform raw reviews into clean DataFrame"""
        try:
            df = pd.DataFrame(raw_reviews)
            
            # Select and rename columns
            df = df[['content', 'score', 'at']].rename(columns={
                'content': 'review',
                'score': 'rating',
                'at': 'date'
            })
            
            # Add metadata
            df['bank'] = bank_name
            df['source'] = 'Google Play'
            
            # Generate IDs
            df['review_id'] = df.apply(
                lambda x: self._generate_review_id(x['review'], str(x['date'])), 
                axis=1
            )
            
            # Clean data
            df = df[df['review'].str.len() >= PREPROCESSING_CONFIG["min_review_length"]]
            df['date'] = pd.to_datetime(df['date']).dt.strftime(PREPROCESSING_CONFIG["date_format"])
            df['rating'] = df['rating'].astype(int)
            
            return df[PREPROCESSING_CONFIG["output_columns"]]
            
        except Exception as e:
            self.logger.error(f"Cleaning failed: {str(e)}")
            raise