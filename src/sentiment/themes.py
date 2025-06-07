import spacy
import pandas as pd
from collections import defaultdict
from typing import List, Dict
import logging

class ThemeAnalyzer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        try:
            self.nlp = spacy.load("en_core_web_sm")
            self.logger.info("SpaCy model loaded successfully.")
        except OSError:
            self.logger.error("SpaCy model not found. Run: python -m spacy download en_core_web_sm")
            raise

    def extract_keywords(self, text: str) -> List[str]:
        """Extract meaningful keywords from text."""
        if text:  # Check if text is not empty
            doc = self.nlp(text)
            return [
                token.lemma_.lower() for token in doc
                if not token.is_stop and not token.is_punct and token.is_alpha and len(token) > 2
            ]
        else:
            self.logger.warning("Empty review text provided.")
            return []

    def identify_themes(self, df: pd.DataFrame, bank_name: str) -> Dict[str, List[str]]:
        """Cluster reviews into themes, separating positive and negative sentiments."""
        # Define theme keywords with positive and negative sentiments
        theme_keywords = {
            "UI/UX": {
                "positive": ["great", "easy", "intuitive", "smooth"],
                "negative": ["bad", "complicated", "difficult", "clunky"]
            },
            "Performance": {
                "positive": ["fast", "quick", "responsive", "efficient"],
                "negative": ["slow", "lag", "crash", "freeze"]
            },
            "Features": {
                "positive": ["useful", "functional", "innovative"],
                "negative": ["missing", "limited", "confusing"]
            },
            "Customer Support": {
                "positive": ["helpful", "responsive", "supportive"],
                "negative": ["unresponsive", "lacking", "poor"]
            },
            "Security": {
                "positive": ["secure", "protected", "trustworthy"],
                "negative": ["vulnerable", "unsafe", "risky"]
            }
        }
        
        theme_counts = defaultdict(lambda: {"positive": 0, "negative": 0})
        theme_examples = defaultdict(lambda: {"positive": [], "negative": []})

        # Process each review
        for _, row in df.iterrows():
            keywords = self.extract_keywords(row['review'])
            matched_themes = self._find_matching_themes(keywords, theme_keywords)
            
            for theme, sentiment in matched_themes:
                theme_counts[theme][sentiment] += 1
                theme_examples[theme][sentiment].append(row['review'][:100] + "...")  # Store review snippet

        return {
            "bank": bank_name,
            "total_reviews": len(df),
            "themes": {k: dict(v) for k, v in theme_counts.items()},
            "examples": {k: {sent: v[:3] for sent, v in val.items()} for k, val in theme_examples.items()}  # Top 3 examples per theme
        }

    def _find_matching_themes(self, keywords: List[str], theme_keywords: Dict[str, Dict[str, List[str]]]) -> List[tuple]:
        """Helper method to find matching themes for the given keywords, separating sentiment."""
        matched_themes = []
        for theme, sentiments in theme_keywords.items():
            for sentiment, words in sentiments.items():
                if any(word in keywords for word in words):
                    matched_themes.append((theme, sentiment))
        return matched_themes

    def document_grouping_logic(self):
        """Document the logic used for grouping keywords into themes."""
        grouping_logic = {
            "UI/UX": "Focuses on user interface and experience-related feedback.",
            "Performance": "Covers reviews mentioning speed and reliability.",
            "Features": "Includes requests or comments on app functionalities.",
            "Customer Support": "Relates to user experiences with support services.",
            "Security": "Involves concerns around account security and data protection."
        }
        for theme, description in grouping_logic.items():
            self.logger.info(f"{theme}: {description}")

# Example usage:
# df = pd.DataFrame({'review': ['Great UI and fast transactions.', 'Customer support is lacking.']})
# analyzer = ThemeAnalyzer()
# result = analyzer.identify_themes(df, "Sample Bank")
# print(result)