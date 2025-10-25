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
    Loads a batch of cleaned reviews from the database.
    """
    query = f"""
        SELECT
            listing_id,
            review_text,
            review_lang,
            property_country,
            property_city
        FROM
            cleaned_reviews_view
        ORDER BY
            listing_id
        OFFSET {offset} ROWS
        FETCH NEXT {BATCH_SIZE} ROWS ONLY
    """
    df = pd.read_sql(query, engine)

    return df