# Shared configuration across modules
BANKS = {
    "CBE": "com.cbe.mobile",
    "BOA": "com.boa.boaMobileBanking",
    "Dashen": "com.dashen.dashensuperapp"
}

SCRAPER_CONFIG = {
    'max_retries': 3,
    'retry_delay': 2,  # seconds
    'target_reviews': 400
}