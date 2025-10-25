import pyodbc
from src.rag_airbnb_config import SQL_SERVER, DATABASE

def test_db_connection():
    """Tests the connection to the SQL Express database using the project's configuration."""
    conn_str = (
        f"Driver={{ODBC Driver 18 for SQL Server}};"
        f"Server={SQL_SERVER};"
        f"Database={DATABASE};"
        f"Trusted_Connection=yes;"
    )
    try:
        conn = pyodbc.connect(conn_str)
        print("✅ Successfully connected to the database.")
        conn.close()
    except pyodbc.Error as ex:
        print("❌ Error connecting to the database:")
        print(ex)

if __name__ == "__main__":
    test_db_connection()