import pandas as pd
import sys
import os
import pyodbc

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag_airbnb_config import SQL_SERVER, DATABASE

def preprocess_and_clean_data():
    """
    Creates a SQL view that represents the cleaned and preprocessed review data.
    This avoids duplicating data and saves storage space.
    """
    print("Starting data preprocessing and cleaning by creating a SQL view...")
    conn_str = (
        f"Driver={{ODBC Driver 18 for SQL Server}};"
        f"Server={SQL_SERVER};"
        f"Database={DATABASE};"
        f"Trusted_Connection=yes;"
    )
    try:
        cnxn = pyodbc.connect(conn_str)
        cursor = cnxn.cursor()
    except Exception as e:
        print(f"❌ Error connecting to the database: {e}")
        return

    view_name = "cleaned_reviews_view"

    # Drop the view if it already exists to ensure a fresh creation
    drop_view_sql = f"IF OBJECT_ID('{view_name}', 'V') IS NOT NULL DROP VIEW {view_name};"
    try:
        cursor.execute(drop_view_sql)
        cnxn.commit()
        print(f"Dropped existing view '{view_name}' (if it existed).")
    except Exception as e:
        print(f"❌ Error dropping existing view {view_name}: {e}")
        # Continue anyway, as the view might not exist

    # SQL statement to create the view with cleaning logic
    create_view_sql = f"""
        CREATE VIEW {view_name} AS
        SELECT
            fr.listing_id,
            fr.comments AS review_text,
            fr.review_lang,
            dl.property_country,
            dl.property_city AS city
        FROM
            fact_reviews AS fr
        JOIN
            dim_listings AS dl ON fr.listing_id = dl.listing_id
        WHERE
            fr.comments IS NOT NULL
            AND LTRIM(RTRIM(fr.comments)) <> ''
            AND fr.comments NOT LIKE '%unknown%'
            AND dl.property_country IS NOT NULL
            AND LTRIM(RTRIM(dl.property_country)) <> ''
            AND dl.property_country NOT LIKE '%unknown%'
            AND dl.property_city IS NOT NULL
            AND LTRIM(RTRIM(dl.property_city)) <> ''
            AND dl.property_city NOT LIKE '%unknown%'
    """

    try:
        cursor.execute(create_view_sql)
        cnxn.commit()
        print(f"✅ Successfully created view '{view_name}'.")
        print("This view dynamically provides cleaned data without duplicating storage.")
    except Exception as e:
        print(f"❌ Error creating view {view_name}: {e}")
        return
    finally:
        cnxn.close()

if __name__ == "__main__":
    preprocess_and_clean_data()