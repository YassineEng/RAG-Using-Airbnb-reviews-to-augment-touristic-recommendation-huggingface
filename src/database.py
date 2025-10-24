import pandas as pd
from sqlalchemy import create_engine, text
from src.config import SQL_SERVER, DATABASE, BATCH_SIZE

def get_db_engine():
    """Initializes and returns a SQLAlchemy engine for the SQL Express database."""
    conn_str = f"mssql+pyodbc://@{SQL_SERVER}/{DATABASE}?driver=ODBC+Driver+17+for+SQL+Server"
    try:
        engine = create_engine(conn_str)
        # Test connection
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        print("✅ Database engine created and connection tested successfully.")
        return engine
    except Exception as e:
        print(f"❌ Error creating database engine or testing connection: {e}")
        return None

def load_reviews_batch(engine, offset):
    """
    Loads a batch of reviews from the database and filters out problematic entries.
    Filters:
    - review_text is NULL
    - review_text is empty or whitespace only
    - review_text contains 'unknown' (case-insensitive)
    """
    query = f"""
        SELECT listing_id, review_text, lang_detect, rating
        FROM fact_reviews
        ORDER BY listing_id
        OFFSET {offset} ROWS
        FETCH NEXT {BATCH_SIZE} ROWS ONLY
    """
    df = pd.read_sql(query, engine)

    # Filter out problematic reviews
    initial_rows = len(df)
    if not df.empty:
        # Filter out NULLs
        df = df.dropna(subset=['review_text'])
        # Filter out empty strings or whitespace only
        df = df[df['review_text'].str.strip().astype(bool)]
        # Filter out 'unknown' (case-insensitive)
        df = df[~df['review_text'].str.contains('unknown', case=False, na=False)]

    filtered_rows = len(df)
    if initial_rows > filtered_rows:
        print(f"Filtered {initial_rows - filtered_rows} problematic reviews from the batch.")

    return df