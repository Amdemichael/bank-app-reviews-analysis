# Bank App Reviews Analysis

## Project Overview
This project analyzes customer satisfaction with mobile banking apps in Ethiopia by scraping user reviews from the Google Play Store for three banks: Commercial Bank of Ethiopia (CBE), Bank of Abyssinia (BOA), and Dashen Bank. The goal is to extract insights regarding user sentiment, identify key themes, and provide actionable recommendations for app improvements.

## Folder Structure

BANK-APP-REVIEWS-ANALYSIS/
├── .github/
├── config/
│ ├── settings.py
│ └── constants.py
├── notebooks/
│ └── data_collection.ipynb
├── data/
│ └── processed/
│ ├── all_banks_combined.csv
│ └── all_banks_cleaned.csv
└── src/
├── data_collection/
│ ├── init.py
│ └── scraper.py
│ └── preprocessor.py


## Prerequisites
- Python 3.10.x
- Required libraries listed in `requirements.txt`

## Installation
1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd BANK-APP-REVIEWS-ANALYSIS
   ```

## Installation

To get started, clone this repository and install the required packages:

```bash
git clone https://github.com/Amdemichael/bank-app-reviews-analysis.git
cd bank-app-reviews-analysis
pip install -r requirements.txt
```

Usage
Task 1: Data Collection and Preprocessing
1 Scraping Reviews:
- Run the scraper.py script to scrape reviews from the Google Play Store.
```python
  python src/data_collection/scraper.py
 ```

This will create individual and combined CSV file of reviews for all banks.

2. Preprocessing Reviews:
- Run the preprocessor.py script to clean the scraped data.
```python
python src/data_collection/preprocessor.py
```
- This will generate a cleaned CSV file with the necessary columns for analysis.
We scraped 1,200+ user reviews from:
- Commercial Bank of Ethiopia
- Bank of Abyssinia
- Dashen Bank

Reviews were cleaned and saved in CSV format.