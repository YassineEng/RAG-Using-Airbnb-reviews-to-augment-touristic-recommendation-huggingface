
import pyodbc

def test_sql_express_connection():
    """Tests connection to a SQL Express database."""
    # IMPORTANT: Replace 'your_database_name' with your actual database name.
    # If using SQL Server Authentication, replace 'Trusted_Connection=yes'
    # with 'UID=your_username;PWD=your_password'
    conn_str = (
        "DRIVER={ODBC Driver 17 for SQL Server}";
        "SERVER=localhost\SQLEXPRESS";
        "DATABASE=your_database_name";
        "Trusted_Connection=yes";
    )
    try:
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        result = cursor.fetchone()
        if result[0] == 1:
            print("SQL Express database connection successful!")
        else:
            print("SQL Express database connection failed.")
        conn.close()
    except pyodbc.Error as e:
        print(f"SQL Express connection error: {e}")
        print("Please ensure 'your_database_name' is correct and the SQL Express instance is running.")
        print("If using SQL Server Authentication, update UID and PWD in the connection string.")


def test_database_connection():
    """
    Tests a database connection.
    Modify this function to connect to your specific database.
    """
    print("--- Testing SQLite In-Memory Database ---")
    test_sqlite_connection()

    print("\n--- Testing SQL Express Database ---")
    test_sql_express_connection()

    # --- Example for PostgreSQL (uncomment and modify as needed) ---
    # import psycopg2
    # try:
    #     conn = psycopg2.connect(
    #         host="your_host",
    #         database="your_database",
    #         user="your_user",
    #         password="your_password"
    #     )
    #     cursor = conn.cursor()
    #     cursor.execute("SELECT 1")
    #     print("PostgreSQL connection successful!")
    #     conn.close()
    # except psycopg2.Error as e:
    #     print(f"PostgreSQL connection error: {e}")

    # --- Example for MySQL (uncomment and modify as needed) ---
    # import mysql.connector
    # try:
    #     conn = mysql.connector.connect(
    #         host="your_host",
    #         database="your_database",
    #         user="your_user",
    #         password="your_password"
    #     )
    #     cursor = conn.cursor()
    #     cursor.execute("SELECT 1")
    #     print("MySQL connection successful!")
    #     conn.close()
    # except mysql.connector.Error as e:
    #     print(f"MySQL connection error: {e}")

    # --- Example for MongoDB (uncomment and modify as needed) ---
    # from pymongo import MongoClient
    # try:
    #     client = MongoClient("mongodb://localhost:27017/")
    #     # The ismaster command is cheap and does not require auth.
    #     client.admin.command('ismaster')
    #     print("MongoDB connection successful!")
    #     client.close()
    # except Exception as e:
    #     print(f"MongoDB connection error: {e}")


if __name__ == "__main__":
    test_database_connection()