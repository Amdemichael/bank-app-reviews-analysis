import cx_Oracle

BANKS = [
    "Commercial Bank of Ethiopia",
    "Bank of Abyssinia",
    "Dashen Bank"
]

dsn = cx_Oracle.makedsn("localhost", 1521, service_name="XEPDB1")
conn = cx_Oracle.connect(user="BANK_REVIEWS", password="Mire#123", dsn=dsn)
cursor = conn.cursor()

for bank_name in BANKS:
    try:
        cursor.execute("""
            MERGE INTO banks b
            USING (SELECT :bank_name AS bank_name FROM dual) input
            ON (b.bank_name = input.bank_name)
            WHEN NOT MATCHED THEN
                INSERT (bank_name)
                VALUES (input.bank_name)
        """, bank_name=bank_name)
        print(f"✅ Inserted or skipped: {bank_name}")
    except cx_Oracle.Error as e:
        print(f"❌ Error inserting {bank_name}: {e}")

conn.commit()
cursor.close()
conn.close()

