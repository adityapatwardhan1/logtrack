import sys
import os
import argparse
from core.parser import parse_log_line
from db.init_db import get_db_connection


def ingest_log_file(file_path, db_path="logtrack.db"):
    """
    Inserts log entries from file at location specified by file_path into DB at db_path.

    :param file_path: Path to the log file to be ingested
    :type file_path: str
    :param db_path: Path to SQLite database file
    :type db_path: str
    """
    if not os.path.exists(file_path):
        print(f'Could not find log file {file_path}, exiting...')
        sys.exit(1)

    con = get_db_connection(db_path)
    cur = con.cursor()

    entries_to_insert = []
    with open(file_path, 'r') as log_file:
        for line in log_file:
            line = line.strip()
            if not line:
                continue

            parsed_line = parse_log_line(line)
            fields = [parsed_line['timestamp'], parsed_line['service'],
                      parsed_line['message'], parsed_line['user']]
            entries_to_insert.append(fields)

    cur.executemany(
        'INSERT INTO logs (timestamp, service, message, user) VALUES (?, ?, ?, ?)',
        entries_to_insert
    )

    con.commit()
    con.close()

    print(f'Successfully inserted {len(entries_to_insert)} entries into {file_path}')


def main():
    """Main logic for CLI application"""
    parser = argparse.ArgumentParser(description='Ingest logs into LogTrack database.')
    parser.add_argument('log_file', help='Path to the log file to ingest.')
    parser.add_argument('--db-path', default='logtrack.db',
                        help='Path to SQLite database file (default: logtrack.db)')
    args = parser.parse_args()

    ingest_log_file(args.log_file, db_path=args.db_path)


# To run: 
# cd logtrack/ 
# python -m cli.ingest_logs sample.log
if __name__ == '__main__':
    main()