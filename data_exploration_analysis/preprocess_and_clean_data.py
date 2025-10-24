import pandas as pd
from sqlalchemy import create_engine, text
import sys
import os

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import SQL_SERVER, DATABASE
from src.database import get_db_engine

def preprocess_and_clean_data():
    """
    Loads review and listing data, cleans it, and saves it to a new table.
    """
    print("Starting data preprocessing and cleaning...")
    engine = get_db_engine()
    if not engine:
        print("❌ Failed to get database engine. Exiting.")
        return

    # Load data
    print("Loading data from fact_reviews and dim_listings...")
    try:
        reviews_df = pd.read_sql("SELECT listing_id, comments, review_lang FROM fact_reviews", engine)
        listings_df = pd.read_sql("SELECT listing_id, property_country, property_city FROM dim_listings", engine)
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    print(f"Loaded {len(reviews_df)} reviews and {len(listings_df)} listings.")

    # Merge data
    df = pd.merge(reviews_df, listings_df, on="listing_id")
    print(f"Merged data into a single DataFrame with {len(df)} rows.")

    # Cleaning logic
    print("Applying cleaning logic...")
    initial_rows = len(df)
    if not df.empty:
        # Filter out NULLs, empty strings or 'unknown' for comments
        df = df.dropna(subset=['comments'])
        df = df[df['comments'].str.strip().astype(bool)]
        df = df[~df['comments'].str.contains('unknown', case=False, na=False)]

        # Filter out NULLs or empty strings for property_country
        df = df.dropna(subset=['property_country'])
        df = df[df['property_country'].str.strip().astype(bool)]

        # Filter out NULLs or empty strings for property_city
        df = df.dropna(subset=['property_city'])
        df = df[df['property_city'].str.strip().astype(bool)]

    filtered_rows = len(df)
    if initial_rows > filtered_rows:
        print(f"Filtered {initial_rows - filtered_rows} problematic rows.")

    # Rename columns to match what load_reviews_batch was producing
    df = df.rename(columns={"comments": "review_text"})

    # Save cleaned data to a new table
    print("Saving cleaned data to 'cleaned_reviews' table...")
    try:
        df.to_sql("cleaned_reviews", engine, if_exists="replace", index=False)
        print("✅ Cleaned data saved successfully.")
    except Exception as e:
        print(f"❌ Error saving cleaned data: {e}")
        return

if __name__ == "__main__":
    preprocess_and_clean_data()