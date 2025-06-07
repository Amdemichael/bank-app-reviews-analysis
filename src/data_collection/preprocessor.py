# scripts/preprocessor.py
import pandas as pd
from datetime import datetime
from config.settings import PREPROCESSING_CONFIG, BANKS
import hashlib
import logging
from typing import List, Dict
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReviewPreprocessor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def _generate_review_id(self, text: str, date: str) -> str:
        return hashlib.md5(f"{text}{date}".encode()).hexdigest()

    def clean_data(self, raw_reviews: List[Dict], bank_name: str) -> pd.DataFrame:
        try:
            df = pd.DataFrame(raw_reviews)
            if df.empty:
                logger.warning(f"No data to process for {bank_name}")
                return pd.DataFrame(columns=PREPROCESSING_CONFIG["output_columns"])
            df = df[['content', 'score', 'at']].rename(columns={
                'content': 'review',
                'score': 'rating',
                'at': 'date'
            })
            df['bank'] = bank_name
            df['source'] = 'Google Play'
            df['review_id'] = df.apply(
                lambda x: self._generate_review_id(x['review'], str(x['date'])), axis=1
            )
            # df = df.dropna(subset=['review', 'rating'])
            df = df.drop_duplicates(subset=['review', 'date'], keep='first')
            df = df[df['review'].str.len() >= PREPROCESSING_CONFIG["min_review_length"]]
            df['date'] = pd.to_datetime(df['date']).dt.strftime(PREPROCESSING_CONFIG["date_format"])
            df['rating'] = df['rating'].astype(int)
            return df[PREPROCESSING_CONFIG["output_columns"]]
        except Exception as e:
            logger.error(f"Cleaning failed for {bank_name}: {str(e)}")
            raise

    def process_all_banks(self):
        combined_df = pd.DataFrame()
        for bank_name in BANKS.keys():
            raw_file = f"data/processed/{bank_name.replace(' ', '_').lower()}_raw.csv"
            if Path(raw_file).exists():
                df = pd.read_csv(raw_file)
                cleaned_df = self.clean_data(df.to_dict('records'), bank_name)
                cleaned_df.to_csv(f"data/processed/{bank_name.replace(' ', '_').lower()}_clean.csv", index=False)
                combined_df = pd.concat([combined_df, cleaned_df], ignore_index=True)
            else:
                logger.warning(f"Raw file not found for {bank_name}")
        if not combined_df.empty:
            combined_df.to_csv("data/processed/all_banks_combined.csv", index=False)
            logger.info(f"Processed {len(combined_df)} reviews in total")
        else:
            logger.error("No data processed")

if __name__ == "__main__":
    preprocessor = ReviewPreprocessor()
    preprocessor.process_all_banks()