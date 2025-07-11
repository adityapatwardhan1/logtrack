import json
from datetime import datetime, timedelta
from db.init_db import get_db_connection


def evaluate_rules(db_path: str, rules_path: str) -> list[dict]:
    """
    Applies all detection rules on the logs in the database.

    :param db_path: Path to the SQLite database file
    :param rules_path: Path to the rules_config.json file
    :return: List of alerts triggered (as dictionaries)
    """
    with open(rules_path, 'r') as rules_file:
        rules = json.load(rules_file)

    con = get_db_connection(db_path)
    cur = con.cursor()

    triggered_alerts = []

    for rule in rules:
        service = rule["service"]
        keyword = rule["keyword"]
        threshold = rule["threshold"]
        window_minutes = rule["window_minutes"]

        # Safely query logs matching service and keyword
        cur.execute(
            "SELECT id, timestamp, message FROM logs WHERE service = ? AND message LIKE ?",
            (service, f"%{keyword}%")
        )
        entries = cur.fetchall()

        # Parse timestamps and sort
        parsed_entries = []
        for row in entries:
            try:
                timestamp = datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M:%S")
                parsed_entries.append({
                    "id": row["id"],
                    "timestamp": timestamp,
                    "message": row["message"]
                })
            except Exception as e:
                continue  # Skip bad timestamps

        parsed_entries.sort(key=lambda x: x["timestamp"])

        # Sliding window logic
        for i in range(len(parsed_entries)):
            window_start = parsed_entries[i]["timestamp"]
            window_end = window_start + timedelta(minutes=window_minutes)

            window_entries = [
                e for e in parsed_entries
                if window_start <= e["timestamp"] <= window_end
            ]

            if len(window_entries) >= threshold:
                alert = {
                    "rule_id": rule["id"],
                    "triggered_at": window_entries[-1]["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
                    "message": f"Rule {rule['id']} triggered with {len(window_entries)} matches",
                    "related_log_ids": [e["id"] for e in window_entries]
                }
                triggered_alerts.append(alert)
                break  # Only one alert per rule (for now)

    con.close()
    return triggered_alerts


# import json
# import sqlite3
# from datetime import datetime, timedelta
# from functools import cmp_to_key
# from db.init_db import get_db_connection

# def compare_timestamps(first_entry, second_entry):
#     first = first_entry[1]
#     second = second_entry[1]
#     first = datetime.strptime(first, "%Y-%m-%d %H:%M:%S")
#     second = datetime.strptime(first, "%Y-%m-%d %H:%M:%S")
#     return first < second

# def evaluate_rules(db_path: str, rules_path: str) -> list[dict]:
#     """
#     Applies all detection rules on the logs in the database.

#     :param db_path: Path to the SQLite database file
#     :param rules_path: Path to the rules_config.json file
#     :return: List of alerts triggered (as dictionaries)
#     """

#     # Get rules
#     with open(rules_path, 'r') as rules_file:
#         rules = json.load(rules_file)
    
#     # Get connection to database
#     con = get_db_connection(db_path=db_path)
#     cur = con.cursor()

#     triggered_alerts = []
#     relevant_log_entries = []

#     for rule in rules:
#         cur.execute(f'SELECT * FROM logs WHERE service={rule["service"]} AND message LIKE {rule["keyword"]}')
#         relevant_log_entries.append(cur.fetchall())

#     relevant_log_entries = list(sorted(relevant_log_entries, key=cmp_to_key(compare_timestamps)))
    
