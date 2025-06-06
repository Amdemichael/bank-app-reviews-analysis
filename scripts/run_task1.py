import logging
from src.data_collection.scraper import BankReviewScraper
from src.data_collection.preprocessor import ReviewPreprocessor
from config.settings import BANKS
import pandas as pd
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    scraper = BankReviewScraper()
    preprocessor = ReviewPreprocessor()
    all_reviews = []
    
    # Ensure data directory exists
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)

    for bank_name, app_id in BANKS.items():
        try:
            logger.info(f"Processing {bank_name}...")
            
            # Scrape reviews
            reviews = scraper.scrape_bank(app_id, bank_name)
            
            # Clean and save
            clean_df = preprocessor.clean_data(reviews, bank_name)
            clean_df.to_csv(f"data/processed/{bank_name.lower().replace(' ', '_')}_clean.csv", index=False)
            
            # Save raw data for reference
            pd.DataFrame(reviews).to_csv(f"data/raw/{bank_name.lower().replace(' ', '_')}_raw.json", index=False)
            
            all_reviews.append(clean_df)
            logger.info(f"Completed {bank_name} with {len(clean_df)} reviews")
            
        except Exception as e:
            logger.error(f"Failed to process {bank_name}: {str(e)}")
            continue
    
    if all_reviews:
        combined = pd.concat(all_reviews)
        combined.to_csv("data/processed/all_banks_combined.csv", index=False)
        logger.info(f"Saved combined data with {len(combined)} total reviews")

if __name__ == "__main__":
    main()