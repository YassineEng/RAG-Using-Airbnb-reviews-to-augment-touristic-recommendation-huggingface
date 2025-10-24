import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pyodbc
from src.config import SQL_SERVER, DATABASE

def get_table_schema(table_name):
    conn_str = (
        f"DRIVER={{ODBC Driver 17 for SQL Server}};".replace('{{', '{').replace('}}', '}') + 
        f"SERVER={SQL_SERVER};".replace('{{', '{').replace('}}', '}') + 
        f"DATABASE={DATABASE};".replace('{{', '{').replace('}}', '}') + 
        f"Trusted_Connection=yes;".replace('{{', '{').replace('}}', '}')
    )
    try:
        cnxn = pyodbc.connect(conn_str)
        cursor = cnxn.cursor()

        query = (
            "SELECT COLUMN_NAME, DATA_TYPE "
            "FROM INFORMATION_SCHEMA.COLUMNS "
            "WHERE TABLE_NAME = ? "
            "ORDER BY ORDINAL_POSITION"
        )
        cursor.execute(query, table_name)

        print(f"\nSchema for table '{table_name}' in database '{DATABASE}':")
        columns = cursor.fetchall()
        if columns:
            for col_name, data_type in columns:
                print(f"- {col_name} ({data_type})")
        else:
            print(f"Table '{table_name}' not found or has no columns.")

        cnxn.close()
    except pyodbc.Error as e:
        print(f"❌ Error connecting to or querying the database: {e}")
        print("Please ensure SQL Server is running and connection details in src/config.py are correct.")

import sys

if __name__ == "__main__":
    if len(sys.argv) > 1:
        get_table_schema(sys.argv[1])
    else:
        print("Usage: python get_table_schema.py <table_name>")