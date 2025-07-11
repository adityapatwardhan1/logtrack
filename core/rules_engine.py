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

        parsed = []
        for row in entries:
            try:
                ts = datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M:%S")
                parsed.append({
                    "id": row["id"],
                    "timestamp": ts,
                    "message": row["message"]
                })
            except Exception:
                continue

        parsed.sort(key=lambda x: x["timestamp"])

        # Two-pointer sliding window
        left = 0
        for right in range(len(parsed)):
            while parsed[right]["timestamp"] - parsed[left]["timestamp"] > timedelta(minutes=window_minutes):
                left += 1
            window_size = right - left + 1
            if window_size >= threshold:
                window_slice = parsed[left:right + 1]
                alert = {
                    "rule_id": rule["id"],
                    "triggered_at": parsed[right]["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
                    "message": f"Rule {rule['id']} triggered with {window_size} matches",
                    "related_log_ids": [e["id"] for e in window_slice]
                }
                triggered_alerts.append(alert)
                break  # Only one alert per rule

    con.close()
    return triggered_alerts
