import logging
import pandas as pd
from src.analysis.sentiment import SentimentAnalyzer
from src.analysis.themes import ThemeAnalyzer
from pathlib import Path
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Load cleaned data
    input_path = "data/processed/all_banks_combined.csv"
    output_dir = Path("data/outputs")
    output_dir.mkdir(exist_ok=True)
    
    try:
        df = pd.read_csv(input_path)
        logger.info(f"Loaded {len(df)} reviews for analysis")
    except Exception as e:
        logger.error(f"Failed to load data: {str(e)}")
        return

    # Sentiment Analysis
    logger.info("Starting sentiment analysis...")
    sentiment_analyzer = SentimentAnalyzer()
    df = sentiment_analyzer.analyze_batch(df)
    df.to_csv(output_dir / "reviews_with_sentiment.csv", index=False)
    logger.info("Completed sentiment analysis")

    # Thematic Analysis
    logger.info("Starting thematic analysis...")
    theme_analyzer = ThemeAnalyzer()
    results = []
    
    for bank_name in df['bank'].unique():
        bank_df = df[df['bank'] == bank_name]
        analysis = theme_analyzer.identify_themes(bank_df, bank_name)
        results.append(analysis)
        logger.info(f"Completed {bank_name} thematic analysis")
    
    # Save results
    with open(output_dir / "thematic_analysis.json", "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info("Analysis complete. Results saved to data/outputs/")

if __name__ == "__main__":
    main()