import cx_Oracle
import pandas as pd
from datetime import datetime

# Read CSV
df = pd.read_csv("../notebooks/data/processed/all_banks_combined.csv")


# Prepare Oracle connection
dsn = cx_Oracle.makedsn("localhost", 1521, service_name="XEPDB1")
conn = cx_Oracle.connect(user="BANK_REVIEWS", password="Mire#123", dsn=dsn)
cursor = conn.cursor()

insert_sql = """
    MERGE INTO reviews r
    USING (SELECT :review_id AS review_id FROM dual) input
    ON (r.review_id = input.review_id)
    WHEN NOT MATCHED THEN
      INSERT (review_id, review, rating, review_date, bank_name, source)
      VALUES (:review_id, :review, :rating, :review_date, :bank_name, :source)
"""

for index, row in df.iterrows():
    try:
        # Convert string date to datetime object
        review_date = datetime.strptime(row['date'], '%Y-%m-%d')

        cursor.execute(insert_sql, {
            "review_id": row['review_id'],
            "review": row['review'],
            "rating": int(row['rating']),
            "review_date": review_date,
            "bank_name": row['bank'],
            "source": row['source']
        })
    except cx_Oracle.Error as e:
        print(f"Error inserting review_id {row['review_id']}: {e}")

conn.commit()
cursor.close()
conn.close()
