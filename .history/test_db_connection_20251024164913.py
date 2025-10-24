
import pyodbc

def test_sql_express_connection():
    """Tests connection to a SQL Express database."""
    # IMPORTANT: Replace 'your_database_name' with your actual database name.
    # If using SQL Server Authentication, replace 'Trusted_Connection=yes'
    # with 'UID=your_username;PWD=your_password'
    conn_str = (
        "DRIVER={ODBC Driver 17 for SQL Server}";
        "SERVER=YASSINE\SQLEXPRESS";
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

