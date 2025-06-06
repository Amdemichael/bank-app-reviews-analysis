from google_play_scraper import reviews, Sort
import time
from tqdm import tqdm
from typing import List, Dict
from config.settings import BANKS, SCRAPER_CONFIG
from pathlib import Path
import logging

class BankReviewScraper:
    def __init__(self):
        self.sort_method = getattr(Sort, SCRAPER_CONFIG["sort_method"])
        self.logger = logging.getLogger(__name__)

    def _scrape_batch(self, app_id: str, continuation_token: str = None) -> tuple:
        """Scrape a single batch of reviews"""
        return reviews(
            app_id,
            lang=SCRAPER_CONFIG["language"],
            country=SCRAPER_CONFIG["country"],
            sort=self.sort_method,
            count=SCRAPER_CONFIG["batch_size"],
            continuation_token=continuation_token
        )

    def scrape_bank(self, app_id: str, bank_name: str) -> List[Dict]:
        """Scrape all reviews for a single bank"""
        results = []
        continuation_token = None
        
        with tqdm(total=SCRAPER_CONFIG["target_reviews"], desc=f"Scraping {bank_name}") as pbar:
            for attempt in range(SCRAPER_CONFIG["max_retries"]):
                try:
                    batch, continuation_token = self._scrape_batch(app_id, continuation_token)
                    results.extend(batch)
                    pbar.update(len(batch))
                    
                    if len(results) >= SCRAPER_CONFIG["target_reviews"] or not continuation_token:
                        break
                        
                    time.sleep(SCRAPER_CONFIG["delay_seconds"])
                    
                except Exception as e:
                    self.logger.error(f"Attempt {attempt + 1} failed: {str(e)}")
                    time.sleep(SCRAPER_CONFIG["delay_seconds"] * 2)
        
        return results[:SCRAPER_CONFIG["target_reviews"]]