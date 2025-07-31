import sys
import os
import argparse
import json
import csv
from datetime import datetime
from core.parser import parse_log_line
from db.init_db import get_db_connection


def parse_flexible_timestamp(raw_ts):
    """Try multiple timestamp formats, return formatted timestamp or raise."""
    formats = [
        "%y%m%d %H%M%S",         # e.g. '250711 130000'
        "%d/%m/%Y %H:%M:%S",     # e.g. '01/01/2023 13:00:00'
        "%Y-%m-%d %H:%M:%S",     # ISO-like with space
        "%m/%d/%Y %H:%M:%S",     # US-style
        "%Y-%m-%dT%H:%M:%S"      # ISO 8601 with 'T'
    ]
    for fmt in formats:
        try:
            return datetime.strptime(raw_ts, fmt).strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue
    # Final fallback: try Python's ISO parser directly
    try:
        return datetime.fromisoformat(raw_ts).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        raise ValueError(f"Unrecognized timestamp format: {raw_ts}")



def ingest_log_file(file_path, db_path="logtrack.db"):
    if not os.path.exists(file_path):
        print(f'Could not find log file {file_path}, exiting...')
        sys.exit(1)

    con = get_db_connection(db_path)
    cur = con.cursor()
    entries_to_insert = []

    if file_path.endswith(".json"):
        with open(file_path, 'r') as log_file:
            try:
                data = json.load(log_file)
                if isinstance(data, dict):
                    data = [data]
            except json.JSONDecodeError:
                log_file.seek(0)
                data = []
                for line in log_file:
                    line = line.strip()
                    if line:
                        try:
                            data.append(json.loads(line))
                        except json.JSONDecodeError:
                            print(f"Skipping malformed JSON line: {line}")

        for entry in data:
            try:
                raw_ts = entry['timestamp']
                timestamp = parse_flexible_timestamp(raw_ts)
                service = entry['service']
                message = entry['message']
                user = entry.get('user', 'unknown') or 'unknown'
                entries_to_insert.append([timestamp, service, message, user])
            except KeyError as e:
                print(f"Skipping entry missing field {e}: {entry}")

    elif file_path.endswith(".csv"):
        with open(file_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    raw_ts = f"{row.get('Date', '').strip()} {row.get('Time', '').strip()}".strip()
                    if not raw_ts or raw_ts == " ":
                        raise ValueError("Missing timestamp fields")
                    timestamp = parse_flexible_timestamp(raw_ts)
                    service = row.get('Component', 'unknown').strip()
                    message = row.get('Content', '').strip()
                    user = row.get('User', 'unknown').strip() or 'unknown'
                    entries_to_insert.append([timestamp, service, message, user])
                except Exception as e:
                    print(f"Skipping CSV row due to error: {e}")

    else:
        with open(file_path, 'r') as log_file:
            for line in log_file:
                line = line.strip()
                if not line:
                    continue
                parsed_line = parse_log_line(line)
                fields = [
                    parsed_line['timestamp'],
                    parsed_line['service'],
                    parsed_line['message'],
                    parsed_line['user']
                ]
                entries_to_insert.append(fields)

    cur.executemany(
        'INSERT INTO logs (timestamp, service, message, user) VALUES (?, ?, ?, ?)',
        entries_to_insert
    )
    con.commit()
    con.close()

    print(f'Successfully inserted {len(entries_to_insert)} entries from {file_path} into {db_path}')


def main():
    parser = argparse.ArgumentParser(description='Ingest logs into LogTrack database.')
    parser.add_argument('log_file', help='Path to the log file to ingest.')
    parser.add_argument('--db-path', default='logtrack.db',
                        help='Path to SQLite database file (default: logtrack.db)')
    args = parser.parse_args()
    ingest_log_file(args.log_file, db_path=args.db_path)


if __name__ == '__main__':
    main()
