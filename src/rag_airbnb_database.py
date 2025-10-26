# This script handles the connection to the primary database (e.g., SQL Server)
# and provides functions to load the Airbnb review data.

import pyodbc
from src.rag_airbnb_config import SQL_SERVER, DATABASE, TABLE, MDF_FILE_PATH, ODBC_DRIVER

def load_reviews(limit: int = 0):
    """Loads reviews from the specified SQL Server database.

    This function establishes a connection to the database using the configuration
    provided in `rag_airbnb_config.py`. It fetches the review data, including
    review_id, listing_id, and the review text (comments).

    Args:
        limit (int): The maximum number of reviews to load. If set to 0, all reviews
                     will be loaded.

    Returns:
        list[dict]: A list of dictionaries, where each dictionary represents a review
                    with keys 'review_id', 'listing_id', and 'text'. Returns an
                    empty list if an error occurs.
    """
    # Construct the connection string for the SQL Server database.
    # This uses trusted (Windows Authentication) connection and attaches the database file directly.
    conn_str = (
        f"Driver={ODBC_DRIVER};"
        f"Server={SQL_SERVER};"
        f"Trusted_Connection=yes;"
        f"AttachDbFilename={MDF_FILE_PATH};"
        f"DATABASE={DATABASE};"
    )
    try:
        # Establish the database connection.
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()

        # Construct the SQL query to select the reviews.
        # If a limit is specified, a TOP clause is added to the query.
        query_top_clause = f"TOP {limit}" if limit > 0 else ""
        query = f"SELECT {query_top_clause} review_id, listing_id, comments FROM {TABLE} WHERE comments IS NOT NULL;"

        # Execute the query and fetch all results.
        rows = cursor.execute(query).fetchall()
        conn.close()

        # Process the fetched rows into a list of dictionaries.
        # The review_id and listing_id are converted to strings, and only reviews with non-empty
        # comments are included.
        reviews = [{"review_id": str(r[0]), "listing_id": str(r[1]), "text": r[2]} for r in rows if r[2]]
        print(f"[+] Loaded {len(reviews)} reviews.")
        return reviews
    except Exception as e:
        # Handle any exceptions that occur during the database operation.
        print(f"❌ Error loading reviews: {e}")
        return []
