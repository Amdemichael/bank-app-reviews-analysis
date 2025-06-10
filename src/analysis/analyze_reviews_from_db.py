# analyze_reviews_from_db.py

import pandas as pd
import cx_Oracle
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from nltk.corpus import stopwords
from collections import Counter
import nltk
import re
import os
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.colors import Color
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import spacy

# Initialize NLTK and spaCy
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))
nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
analyzer = SentimentIntensityAnalyzer()

# DB connection
def load_data():
    try:
        dsn = cx_Oracle.makedsn("localhost", 1521, service_name="XEPDB1")
        conn = cx_Oracle.connect(user="BANK_REVIEWS", password=os.getenv("DB_PASSWORD", "Mire#123"), dsn=dsn)
        
        query = """
        SELECT REVIEW_ID, REVIEW, RATING, REVIEW_DATE, BANK_NAME, SOURCE 
        FROM REVIEWS
        """
        
        cursor = conn.cursor()
        cursor.execute(query)
        
        rows = []
        for row in cursor:
            review_id, review_lob, rating, review_date, bank_name, source = row
            review_text = review_lob.read() if review_lob is not None else ""
            rows.append((review_id, review_text, rating, review_date, bank_name, source))
        
        cursor.close()
        conn.close()
        
        df = pd.DataFrame(rows, columns=["REVIEW_ID", "REVIEW", "RATING", "REVIEW_DATE", "BANK_NAME", "SOURCE"])
        print("DataFrame columns:", df.columns.tolist())
        print("DataFrame shape:", df.shape)
        if df.empty:
            print("⚠️ Warning: No data retrieved from the database.")
        return df
    
    except cx_Oracle.Error as e:
        print(f"Database error: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Unexpected error: {e}")
        return pd.DataFrame()

# Add sentiment category and text-based sentiment
def process_data(df):
    if df.empty:
        print("⚠️ Warning: Empty DataFrame passed to process_data.")
        return df
    
    required_columns = ['REVIEW_DATE', 'RATING', 'REVIEW', 'BANK_NAME']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"⚠️ Error: Missing columns in DataFrame: {missing_columns}")
        return df
    
    df['REVIEW_DATE'] = pd.to_datetime(df['REVIEW_DATE'], errors='coerce')
    df['sentiment_rating'] = df['RATING'].apply(lambda r: 'positive' if r >= 4 else 'negative' if r <= 2 else 'neutral')
    df['sentiment_text'] = df['REVIEW'].apply(lambda x: text_sentiment(x) if pd.notna(x) else 'neutral')
    df.columns = [col.lower() for col in df.columns]
    return df

# Text-based sentiment analysis
def text_sentiment(text):
    score = analyzer.polarity_scores(text)['compound']
    return 'positive' if score > 0.05 else 'negative' if score < -0.05 else 'neutral'

# Clean and tokenize with spaCy
def clean_and_tokenize(text):
    if pd.isna(text) or not text:
        return []
    doc = nlp(text.lower())
    return [token.lemma_ for token in doc if token.is_alpha and not token.is_stop and len(token.lemma_) > 2]

# Extract top keywords per sentiment
def extract_keywords(df, sentiment, bank=None, top_n=10):
    if bank:
        df = df[df['bank_name'] == bank]
    tokens = df[df['sentiment_rating'] == sentiment]['review'].dropna().apply(clean_and_tokenize)
    all_words = [word for tokens_list in tokens for word in tokens_list]
    return Counter(all_words).most_common(top_n)

# Check for review bias
def check_bias(df):
    if df.empty:
        return "No data to analyze bias."
    sentiment_counts = df['sentiment_rating'].value_counts(normalize=True)
    bias_report = "\n📉 Sentiment Bias Check:\n"
    bias_report += f"Positive: {sentiment_counts.get('positive', 0):.2%}\n"
    bias_report += f"Neutral: {sentiment_counts.get('neutral', 0):.2%}\n"
    bias_report += f"Negative: {sentiment_counts.get('negative', 0):.2%}\n"
    if sentiment_counts.get('negative', 0) > 0.5:
        bias_report += "⚠️ Warning: Potential negative review bias detected.\n"
    return bias_report

# Plot rating distribution
def plot_rating_distribution(df):
    if df.empty:
        return
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x='rating')
    plt.title("Rating Distribution")
    plt.savefig("outputs/rating_distribution.png")
    plt.close()

# Plot sentiment distribution
def plot_sentiment_distribution(df):
    if df.empty:
        return
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x='sentiment_rating', hue='bank_name')
    plt.title("Sentiment Distribution by Bank")
    plt.savefig("outputs/sentiment_distribution.png")
    plt.close()

# Plot sentiment trend over time
def plot_sentiment_trend(df):
    if df.empty:
        return
    plt.figure(figsize=(10, 6))
    df.groupby([df['review_date'].dt.to_period('M'), 'sentiment_rating']).size().unstack().plot()
    plt.title("Sentiment Trend Over Time")
    plt.savefig("outputs/sentiment_trend.png")
    plt.close()

# Generate word cloud
def generate_wordcloud(df, bank, sentiment):
    reviews = df[(df['bank_name'] == bank) & (df['sentiment_rating'] == sentiment)]['review'].dropna()
    if reviews.empty:
        print(f"No {sentiment} reviews for {bank}")
        return
    tokens = [word for review in reviews for word in clean_and_tokenize(review)]
    if not tokens:
        print(f"No tokens for {bank} - {sentiment}")
        return
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(tokens))
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"{bank} - {sentiment.capitalize()} Word Cloud")
    plt.savefig(f"outputs/{bank.replace(' ', '_')}_{sentiment}_wordcloud.png")
    plt.close()

# Summarize insights
def summarize_insights(df):
    banks = df['bank_name'].unique()
    insights = {}
    for bank in banks:
        top_pos = extract_keywords(df, 'positive', bank)
        top_neg = extract_keywords(df, 'negative', bank)
        insights[bank] = {
            "drivers": top_pos,
            "pain_points": top_neg
        }
    return insights

# Suggest improvements dynamically
def suggest_improvements(insights):
    pain_point_categories = {
        'stability': ['crash', 'bug', 'error', 'fail'],
        'authentication': ['login', 'password', 'access'],
        'performance': ['slow', 'lag', 'delay'],
        'customer_service': ['support', 'service', 'help']
    }
    suggestions = []
    for bank, data in insights.items():
        for category, keywords in pain_point_categories.items():
            if any(word in [w for w, _ in data['pain_points']] for word in keywords):
                suggestions.append(f"💡 {bank}: Improve {category} (e.g., address {', '.join(keywords)}).")
        drivers = [w for w, _ in data['drivers']][:3]
        if drivers:
            suggestions.append(f"✨ {bank}: Promote positive features like {', '.join(drivers)} in marketing.")
    return suggestions

# Generate PDF report with red pain points
def generate_report(df, insights, output_path="outputs/report.pdf"):
    c = canvas.Canvas(output_path, pagesize=letter)
    width, height = letter
    y = height - 50
    
    # Page 1: Title and Bias Check
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, "Bank Reviews Analysis Report")
    y -= 30
    c.setFont("Helvetica", 12)
    c.drawString(50, y, "Generated on: June 10, 2025")
    y -= 30
    c.drawString(50, y, "Bias Analysis:")
    y -= 20
    for line in check_bias(df).split('\n'):
        c.drawString(50, y, line)
        y -= 20
    c.showPage()
    
    # Page 2: Insights
    y = height - 50
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, "Key Drivers and Pain Points")
    y -= 30
    c.setFont("Helvetica", 12)
    for bank, items in insights.items():
        c.drawString(50, y, f"Bank: {bank}")
        y -= 20
        # Drivers in green
        c.setFillColorRGB(0, 0.5, 0)  # Green
        c.drawString(50, y, f"Drivers: {', '.join([w for w, _ in items['drivers']][:5])}")
        y -= 20
        # Pain Points in red
        c.setFillColorRGB(1, 0, 0)  # Red
        c.drawString(50, y, f"Pain Points: {', '.join([w for w, _ in items['pain_points']][:5])}")
        y -= 20
        c.setFillColorRGB(0, 0, 0)  # Reset to black
        y -= 10
        if y < 50:
            c.showPage()
            y = height - 50
    c.showPage()
    
    # Page 3: Recommendations
    y = height - 50
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, "Recommendations")
    y -= 30
    c.setFont("Helvetica", 12)
    recommendations = suggest_improvements(insights)
    for rec in recommendations:
        c.drawString(50, y, rec)
        y -= 20
        if y < 50:
            c.showPage()
            y = height - 50
    c.showPage()
    
    # Page 4: Visualizations
    y = height - 50
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, "Visualizations")
    y -= 30
    c.setFont("Helvetica", 12)
    c.drawString(50, y, "See outputs/ directory for the following plots:")
    y -= 20
    c.drawString(50, y, "- Rating Distribution (rating_distribution.png)")
    y -= 20
    c.drawString(50, y, "- Sentiment Distribution by Bank (sentiment_distribution.png)")
    y -= 20
    c.drawString(50, y, "- Sentiment Trend Over Time (sentiment_trend.png)")
    y -= 20
    c.drawString(50, y, "- Word Clouds per Bank and Sentiment")
    c.showPage()
    
    c.save()

def main():
    os.makedirs("outputs", exist_ok=True)

    print("📥 Loading data...")
    df = load_data()
    if df.empty:
        print("⚠️ No data loaded. Exiting.")
        return
    
    print("🔄 Processing data...")
    df = process_data(df)
    
    print("📉 Checking for bias...")
    print(check_bias(df))
    
    print("📊 Generating visualizations...")
    plot_rating_distribution(df)
    plot_sentiment_distribution(df)
    plot_sentiment_trend(df)
    for bank in df['bank_name'].unique():
        for sentiment in ['positive', 'negative']:
            generate_wordcloud(df, bank, sentiment)
    
    print("🧠 Analyzing insights...")
    insights = summarize_insights(df)
    
    print("\n📌 Key Drivers and Pain Points:")
    for bank, items in insights.items():
        print(f"\n🏦 {bank}")
        print("✅ Drivers:", [word for word, _ in items['drivers']])
        print("⚠️ Pain Points:", [word for word, _ in items['pain_points']])
    
    print("\n💬 Recommendations:")
    recommendations = suggest_improvements(insights)
    for rec in recommendations:
        print(rec)
    
    print("\n📄 Generating report...")
    generate_report(df, insights)
    print("Report saved to outputs/report.pdf")

if __name__ == "__main__":
    main()