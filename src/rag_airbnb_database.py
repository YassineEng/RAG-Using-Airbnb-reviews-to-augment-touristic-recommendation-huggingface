import pyodbc
from src.rag_airbnb_config import SQL_SERVER, DATABASE, TABLE

def load_reviews(limit: int = 3000):
    """Loads reviews from the SQL Express database."""
    conn_str = (
        f"Driver={{ODBC Driver 18 for SQL Server}};"
        f"Server={SQL_SERVER};"
        f"Database={DATABASE};"
        f"Trusted_Connection=yes;"
    )
    try:
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        query = f"SELECT TOP {limit} review_id, listing_id, comments FROM {TABLE} WHERE comments IS NOT NULL;"
        rows = cursor.execute(query).fetchall()
        conn.close()
        reviews = [{"review_id": str(r[0]), "listing_id": str(r[1]), "text": r[2]} for r in rows if r[2]]
        print(f"[+] Loaded {len(reviews)} reviews.")
        return reviews
    except Exception as e:
        print(f"❌ Error loading reviews: {e}")
        return []