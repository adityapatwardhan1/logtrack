import sqlite3
import os
from pathlib import Path

def init_db():
    try:
        # Read schema
        sql_script = ""
        script_dir = Path(__file__).resolve().parent
        db_path = script_dir.parent / "logtrack.db"
        schema_path = script_dir / "schema.sql"

        with open(schema_path, "r") as schema:
            sql_script = schema.read()

        # Connect to database, execute the commands in the schema
        con = sqlite3.connect(db_path)
        cur = con.cursor()
        cur.executescript(sql_script)
        con.commit()

        # Close connection
        con.close()
        print(f"Initialized database at {db_path} with schema from {schema_path}.")
    
    # Handle exception
    except Exception as e:
        print("An error occurred while initializing the database.")
        print(e)

if __name__ == '__main__':
    init_db()
