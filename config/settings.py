BANKS = {
    "Commercial Bank of Ethiopia": "com.combanketh.mobilebanking",
    "Bank of Abyssinia": "com.boa.boaMobileBanking",
    "Dashen Bank": "com.dashen.dashensuperapp"
}

SCRAPER_CONFIG = {
    "target_reviews": 500,
    "language": "en",
    "country": "et",
    "sort_method": "NEWEST",
    "batch_size": 100,
    "delay_seconds": 2,
    "max_retries": 3
}

PREPROCESSING_CONFIG = {
    "min_review_length": 10,
    "date_format": "%Y-%m-%d",
    "output_columns": ["review_id", "review", "rating", "date", "bank", "source"]
}