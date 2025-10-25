import pandas as pd
import sys
import os
import pyodbc

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag_airbnb_config import SQL_SERVER, DATABASE, MDF_FILE_PATH, ODBC_DRIVER

def generate_data_quality_report():
    """
    Generates a report on data loss due to cleaning by comparing row counts
    before and after the cleaning view.
    """
    print("Generating data quality report...")
    conn_str = (
        f"Driver={ODBC_DRIVER};"
        f"Server={SQL_SERVER};"
        f"Trusted_Connection=yes;"
        f"AttachDbFilename={MDF_FILE_PATH};"
        f"DATABASE={DATABASE};"
    )
    try:
        cnxn = pyodbc.connect(conn_str)
        cursor = cnxn.cursor()
    except Exception as e:
        print(f"❌ Error connecting to the database: {e}")
        return

    # Count initial rows (before cleaning logic)
    initial_rows_query = """
        SELECT COUNT(*)
        FROM fact_reviews AS fr
        JOIN dim_listings AS dl ON fr.listing_id = dl.listing_id
    """
    try:
        cursor.execute(initial_rows_query)
        initial_row_count = cursor.fetchone()[0]
        print(f"✅ Counted initial rows.")
    except Exception as e:
        print(f"❌ Error counting initial rows: {e}")
        cnxn.close()
        return

    # Count rows after cleaning (from the view)
    cleaned_rows_query = "SELECT COUNT(*) FROM cleaned_reviews_view"
    try:
        cursor.execute(cleaned_rows_query)
        cleaned_row_count = cursor.fetchone()[0]
        print(f"✅ Counted cleaned rows from view.")
    except Exception as e:
        print(f"❌ Error counting cleaned rows from view: {e}")
        print("Please ensure 'preprocess_and_clean_data.py' has been run to create the view.")
        cnxn.close()
        return

    print("\n--- Data Quality Report ---")
    print(f"Initial rows (fact_reviews JOIN dim_listings): {initial_row_count}")
    print(f"Cleaned rows (from cleaned_reviews_view): {cleaned_row_count}")

    rows_removed = initial_row_count - cleaned_row_count
    if initial_row_count > 0:
        percentage_removed = (rows_removed / initial_row_count) * 100
        print(f"Rows removed by cleaning: {rows_removed} ({percentage_removed:.2f}%)")
    else:
        print("No initial rows to compare.")
    
    cnxn.close()

if __name__ == "__main__":
    generate_data_quality_report()