BANKS = {
    "Commercial Bank of Ethiopia": "com.combanketh.mobilebanking",
    "Bank of Abyssinia": "com.boa.boaMobileBanking",
    "Dashen Bank": "com.dashen.dashensuperapp"
}

SCRAPER_CONFIG = {
    "target_reviews": 400,
    "language": "en",
    "country": "et",
    "sort_method": "NEWEST",
    "batch_size": 200,
    "delay_seconds": 3,
    "max_retries": 5
}

PREPROCESSING_CONFIG = {
    "min_review_length": 10,
    "date_format": "%Y-%m-%d",
    "output_columns": ["review_id", "review", "rating", "date", "bank", "source"]
}