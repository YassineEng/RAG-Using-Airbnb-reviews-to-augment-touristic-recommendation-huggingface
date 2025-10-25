import pyodbc
from src.rag_airbnb_config import SQL_SERVER, DATABASE, TABLE, MDF_FILE_PATH, ODBC_DRIVER

def load_reviews(limit: int = 0):
    """Loads reviews from the SQL Express database. If limit is 0, loads all reviews."""
    conn_str = (
        f"Driver={ODBC_DRIVER};"
        f"Server={SQL_SERVER};"
        f"Trusted_Connection=yes;"
        f"AttachDbFilename={MDF_FILE_PATH};"
        f"DATABASE={DATABASE};"
    )
    try:
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        
        query_top_clause = f"TOP {limit}" if limit > 0 else ""
        query = f"SELECT {query_top_clause} review_id, listing_id, comments FROM {TABLE} WHERE comments IS NOT NULL;"
        
        rows = cursor.execute(query).fetchall()
        conn.close()
        reviews = [{"review_id": str(r[0]), "listing_id": str(r[1]), "text": r[2]} for r in rows if r[2]]
        print(f"[+] Loaded {len(reviews)} reviews.")
        return reviews
    except Exception as e:
        print(f"❌ Error loading reviews: {e}")
        return []