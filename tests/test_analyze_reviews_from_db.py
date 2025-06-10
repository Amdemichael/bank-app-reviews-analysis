import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import os
from datetime import datetime

# Adjust sys.path to import the module
import sys
sys.path.append(r'D:\Projects\Python\bank-app-reviews-analysis\src\analysis')
import analyze_reviews_from_db

class TestAnalyzeReviewsFromDB(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        # Sample DataFrame for testing
        self.sample_data = {
            "REVIEW_ID": [1, 2, 3],
            "REVIEW": ["Great app!", "Slow and buggy", "Okay service"],
            "RATING": [5, 1, 3],
            "REVIEW_DATE": ["2025-01-01", "2025-02-01", "2025-03-01"],
            "BANK_NAME": ["Bank A", "Bank A", "Bank B"],
            "SOURCE": ["App Store", "Google Play", "App Store"]
        }
        self.df = pd.DataFrame(self.sample_data)
        self.df['REVIEW_DATE'] = pd.to_datetime(self.df['REVIEW_DATE'])

        # Mock spaCy and VADER
        self.mock_nlp = MagicMock()
        self.mock_doc = MagicMock()
        self.mock_token = MagicMock()
        self.mock_token.is_alpha = True
        self.mock_token.is_stop = False
        self.mock_token.lemma_ = "test"
        self.mock_doc.__iter__.return_value = [self.mock_token]
        self.mock_nlp.return_value = self.mock_doc

    @patch('analyze_reviews_from_db.cx_Oracle')
    def test_load_data_success(self, mock_cx_oracle):
        """Test load_data with mocked database connection."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_lob = MagicMock()
        mock_lob.read.return_value = "Great app!"
        mock_cursor.__iter__.return_value = [
            (1, mock_lob, 5, datetime(2025, 1, 1), "Bank A", "App Store")
        ]
        mock_conn.cursor.return_value = mock_cursor
        mock_cx_oracle.connect.return_value = mock_conn
        mock_cx_oracle.makedsn.return_value = "dsn"

        df = analyze_reviews_from_db.load_data()
        self.assertFalse(df.empty)
        self.assertEqual(list(df.columns), ["REVIEW_ID", "REVIEW", "RATING", "REVIEW_DATE", "BANK_NAME", "SOURCE"])
        self.assertEqual(df.iloc[0]['REVIEW'], "Great app!")

    @patch('analyze_reviews_from_db.cx_Oracle')
    def test_load_data_db_error(self, mock_cx_oracle):
        """Test load_data with database connection error."""
        mock_cx_oracle.connect.side_effect = analyze_reviews_from_db.cx_Oracle.Error("Connection failed")
        df = analyze_reviews_from_db.load_data()
        self.assertTrue(df.empty)

    def test_process_data_valid(self):
        """Test process_data with valid DataFrame."""
        df = analyze_reviews_from_db.process_data(self.df.copy())
        self.assertIn('sentiment_rating', df.columns)
        self.assertIn('sentiment_text', df.columns)
        self.assertEqual(df['sentiment_rating'].iloc[0], 'positive')
        self.assertEqual(df['sentiment_rating'].iloc[1], 'negative')
        self.assertEqual(df['sentiment_rating'].iloc[2], 'neutral')
        self.assertTrue(all(col.islower() for col in df.columns))

    def test_process_data_empty(self):
        """Test process_data with empty DataFrame."""
        df = pd.DataFrame()
        result = analyze_reviews_from_db.process_data(df)
        self.assertTrue(result.empty)

    def test_process_data_missing_columns(self):
        """Test process_data with missing columns."""
        df = self.df.drop(columns=['REVIEW_DATE'])
        result = analyze_reviews_from_db.process_data(df)
        self.assertTrue(result.equals(df))  # Should return unchanged

    def test_text_sentiment(self):
        """Test text_sentiment classification."""
        self.assertEqual(analyze_reviews_from_db.text_sentiment("I love this app"), 'positive')
        self.assertEqual(analyze_reviews_from_db.text_sentiment("This is terrible"), 'negative')
        self.assertEqual(analyze_reviews_from_db.text_sentiment("It's okay"), 'neutral')
        self.assertEqual(analyze_reviews_from_db.text_sentiment(""), 'neutral')

    @patch('analyze_reviews_from_db.nlp', new_callable=MagicMock)
    def test_clean_and_tokenize(self, mock_nlp):
        """Test clean_and_tokenize with mocked spaCy."""
        mock_nlp.return_value = self.mock_doc
        tokens = analyze_reviews_from_db.clean_and_tokenize("Test text")
        self.assertEqual(tokens, ["test"])
        self.assertEqual(analyze_reviews_from_db.clean_and_tokenize(""), [])
        self.assertEqual(analyze_reviews_from_db.clean_and_tokenize(None), [])

    def test_extract_keywords(self):
        """Test extract_keywords for positive sentiment."""
        with patch('analyze_reviews_from_db.clean_and_tokenize', return_value=['great', 'app']):
            keywords = analyze_reviews_from_db.extract_keywords(self.df, 'positive', bank='Bank A')
            self.assertEqual(keywords, [('great', 2), ('app', 2)])  # Mocked counts

    def test_check_bias(self):
        """Test check_bias for sentiment distribution."""
        report = analyze_reviews_from_db.check_bias(self.df)
        self.assertIn("Positive: 33.33%", report)
        self.assertIn("Negative: 33.33%", report)
        self.assertIn("Neutral: 33.33%", report)

    def test_check_bias_empty(self):
        """Test check_bias with empty DataFrame."""
        report = analyze_reviews_from_db.check_bias(pd.DataFrame())
        self.assertEqual(report, "No data to analyze bias.")

    def test_summarize_insights(self):
        """Test summarize_insights for drivers and pain points."""
        with patch('analyze_reviews_from_db.extract_keywords') as mock_extract:
            mock_extract.side_effect = [
                [('great', 10), ('app', 5)],  # Positive for Bank A
                [('slow', 8), ('buggy', 3)],  # Negative for Bank A
                [('okay', 6)],  # Positive for Bank B
                [('average', 2)]  # Negative for Bank B
            ]
            insights = analyze_reviews_from_db.summarize_insights(self.df)
            self.assertIn('Bank A', insights)
            self.assertIn('Bank B', insights)
            self.assertEqual(insights['Bank A']['drivers'], [('great', 10), ('app', 5)])
            self.assertEqual(insights['Bank A']['pain_points'], [('slow', 8), ('buggy', 3)])

    def test_suggest_improvements(self):
        """Test suggest_improvements for recommendations."""
        insights = {
            'Bank A': {
                'drivers': [('great', 10), ('app', 5)],
                'pain_points': [('slow', 8), ('buggy', 3)]
            }
        }
        recommendations = analyze_reviews_from_db.suggest_improvements(insights)
        self.assertIn("💡 Bank A: Improve performance (e.g., address slow, lag, delay).", recommendations)
        self.assertIn("💡 Bank A: Improve stability (e.g., address crash, bug, error, fail).", recommendations)
        self.assertIn("✨ Bank A: Promote positive features like great, app in marketing.", recommendations)

    @patch('analyze_reviews_from_db.canvas.Canvas')
    def test_generate_report(self, mock_canvas):
        """Test generate_report with mocked PDF creation."""
        mock_c = MagicMock()
        mock_canvas.return_value = mock_c
        insights = {
            'Bank A': {
                'drivers': [('great', 10), ('app', 5)],
                'pain_points': [('slow', 8), ('buggy', 3)]
            }
        }
        analyze_reviews_from_db.generate_report(self.df, insights, output_path="test_report.pdf")
        mock_c.setFont.assert_any_call("Helvetica-Bold", 16)
        mock_c.drawString.assert_any_call(50, 761, "Bank Reviews Analysis Report")
        mock_c.setFillColorRGB.assert_any_call(1, 0, 0)  # Red for pain points
        mock_c.setFillColorRGB.assert_any_call(0, 0.5, 0)  # Green for drivers
        mock_c.save.assert_called_once()

if __name__ == '__main__':
    unittest.main()