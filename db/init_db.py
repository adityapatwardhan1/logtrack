import sqlite3
import os
from pathlib import Path

def init_db(db_path=None):
    """
    Initializes database from schema.sql, containing logs and alerts tables.
    Defaults to placing logtrack.db in the project root folder.
    """
    try:
        script_dir = Path(__file__).resolve().parent
        schema_path = script_dir / "schema.sql"

        if db_path is None:
            db_path = script_dir.parent / "logtrack.db"

        with open(schema_path, "r") as schema:
            sql_script = schema.read()

        con = sqlite3.connect(db_path)
        cur = con.cursor()
        cur.executescript(sql_script)
        con.commit()
        con.close()

        print(f"Initialized database at {db_path} with schema from {schema_path}.")

    except Exception as e:
        print("An error occurred while initializing the database.")
        print(e)
        raise


def get_db_connection(db_path=None):
    """Gets connection to database at db_path. Initializes DB if missing.
       Defaults to root folder's logtrack.db"""
    if db_path is None:
        script_dir = Path(__file__).resolve().parent
        db_path = script_dir.parent / "logtrack.db"

    if not os.path.exists(db_path):
        print(f"Database file {db_path} not found. Initializing new database...")
        init_db(db_path)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


if __name__ == '__main__':
    init_db()
