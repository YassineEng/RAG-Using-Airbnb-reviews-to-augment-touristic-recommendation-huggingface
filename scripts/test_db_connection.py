import pyodbc
import urllib.parse
from sqlalchemy import text
from sqlalchemy import create_engine


# --- Configuration ---
server_name = r"YASSINE\SQLEXPRESS"  # Your SQL Server instance
mdf_file = r"D:\SQLData\AirbnbDataWarehouse.mdf"  # Path to your .mdf
db_name = "AirbnbDataWarehouse"  # Logical database name

# --- Connection string for pyodbc ---
# Using Trusted Connection (Windows authentication)
conn_str = (
    f"DRIVER={{ODBC Driver 17 for SQL Server}};"
    f"SERVER={server_name};"
    f"Trusted_Connection=yes;"
    f"AttachDbFilename={mdf_file};"
    f"DATABASE={db_name};"
)

# --- Attempt connection ---
try:
    cnxn = pyodbc.connect(conn_str)
    print("✅ Successfully connected to the database.")
except pyodbc.Error as ex:
    print("❌ Error connecting to the database:")
    print(ex)
    print("Check that SQL Server is running, the ODBC driver is installed,")
    print("and that the .mdf file is accessible and not already attached.")
    cnxn = None

# --- Create SQLAlchemy engine for pandas or other libraries ---
if cnxn:
    # URL-encode the connection string for SQLAlchemy
    params = urllib.parse.quote_plus(conn_str)
    engine = create_engine(f"mssql+pyodbc:///?odbc_connect={params}")
    print("✅ SQLAlchemy engine created successfully.")
else:
    engine = None
    print("⚠️ SQLAlchemy engine not created due to connection error.")

# --- Optional: Verify connection by fetching top 5 tables (example) ---
if engine:
    try:
        with engine.connect() as connection:
            result = connection.execute(text("SELECT name FROM sys.tables;"))
            tables = [row[0] for row in result]
            print("Tables in database:", tables)  # print full list
    except Exception as e:
        print("⚠️ Could not fetch tables:", e)