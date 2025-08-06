import pandas as pd
import sqlite3
import os

# --- Configuration ---
CSV_FILE = 'mock_market_data_for_prediction.csv'
DB_FILE = 'market_data.db'
TABLE_NAME = 'electricity_data'


def create_database():
    """
    Reads data from a CSV file and loads it into a SQLite database.
    The function will overwrite the existing table if it already exists.
    """
    if not os.path.exists(CSV_FILE):
        print(f"Error: The file '{CSV_FILE}' was not found.")
        print("Please make sure the CSV file is in the same directory.")
        return

    try:
        # Read the CSV data
        df = pd.read_csv(CSV_FILE)
        print(f"Successfully read {len(df)} rows from {CSV_FILE}.")

        # Ensure the timestamp column is in datetime format for proper sorting
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Connect to the SQLite database (it will be created if it doesn't exist)
        conn = sqlite3.connect(DB_FILE)

        # Write the data to the SQLite table
        # if_exists='replace' will drop the table first if it exists and create a new one.
        df.to_sql(TABLE_NAME, conn, if_exists='replace', index=False)

        conn.close()

        print(f"Database '{DB_FILE}' created successfully.")
        print(f"Data has been loaded into the '{TABLE_NAME}' table.")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == '__main__':
    create_database()
