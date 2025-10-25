import pyodbc
import sys
import os

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag_airbnb_config import SQL_SERVER, DATABASE, MDF_FILE_PATH, ODBC_DRIVER

def test_db_connection():
    """Tests the connection to the SQL Express database using the project's configuration."""
    conn_str = (
        f"Driver={ODBC_DRIVER};"
        f"Server={SQL_SERVER};"
        f"Trusted_Connection=yes;"
        f"AttachDbFilename={MDF_FILE_PATH};"
        f"DATABASE={DATABASE};"
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