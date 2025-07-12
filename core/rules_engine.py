import json
from datetime import datetime, timedelta, timezone
from db.init_db import get_db_connection

def _parse_timestamp(ts_str: str) -> datetime:
    """
    Parses a timestamp string in the format '%Y-%m-%d %H:%M:%S' into a timezone-aware datetime object (UTC).

    :param ts_str: Timestamp string to parse
    :return: Parsed timezone-aware datetime object in UTC
    """
    # Parse naive datetime and set UTC timezone explicitly
    naive_dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
    return naive_dt.replace(tzinfo=timezone.utc)


def find_sliding_windows(events: list[dict], window: timedelta, threshold: int):
    """
    Yields slices of events where the number of events in the time window meets or exceeds the threshold.

    :param events: Sorted list of dicts with 'timestamp' key (timezone-aware datetime objects).
    :param window: Time window size as timedelta.
    :param threshold: Minimum number of events within window to yield.
    :yields: tuple(start_index, end_index) indexes into events list defining a valid window.
    """
    left = 0
    for right in range(len(events)):
        while events[right]["timestamp"] - events[left]["timestamp"] > window:
            left += 1
        if right - left + 1 >= threshold:
            yield left, right


def _keyword_threshold_alerts(cur, rule):
    """
    Detects alerts where the number of logs containing a keyword from a specific service
    exceeds a threshold within a time window.

    :param cur: SQLite database cursor
    :param rule: Rule dictionary containing keys:
                 'service', 'keyword', 'threshold', 'window_minutes', 'id'
    :return: List of triggered alert dictionaries
    """
    service = rule["service"]
    keyword = rule["keyword"]
    threshold = rule["threshold"]
    window = timedelta(minutes=rule["window_minutes"])

    cur.execute(
        "SELECT id, timestamp, message FROM logs WHERE service = ? AND message LIKE ?",
        (service, f"%{keyword}%")
    )
    rows = cur.fetchall()

    parsed = []
    for row in rows:
        try:
            parsed.append({
                "id": row["id"],
                "timestamp": _parse_timestamp(row["timestamp"]),
                "message": row["message"]
            })
        except Exception:
            continue

    parsed.sort(key=lambda x: x["timestamp"])
    alerts = []

    for left, right in find_sliding_windows(parsed, window, threshold):
        alert = {
            "rule_id": rule["id"],
            "triggered_at": parsed[right]["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
            "message": f"Rule {rule['id']} triggered with {right - left + 1} matches",
            "related_log_ids": [e["id"] for e in parsed[left:right + 1]]
        }
        alerts.append(alert)
        break  # Only one alert per rule

    return alerts


# Note: repeated message requires EXACT MESSAGES, 
# oftentimes it's better to use keyword threshold
def _repeated_message_alerts(cur, rule):
    """
    Detects alerts where an exact log message repeats at least a threshold number
    of times within a time window.

    :param cur: SQLite database cursor
    :param rule: Rule dictionary containing keys:
                 'message', 'threshold', 'window_minutes', 'id'
    :return: List of triggered alert dictionaries
    """
    message = rule["message"]
    print("message:",message)
    threshold = rule["threshold"]
    window = timedelta(minutes=rule["window_minutes"])

    cur.execute(
        "SELECT id, timestamp FROM logs WHERE message = ?", (message,)
    )
    rows = cur.fetchall()

    parsed = []
    for row in rows:
        print(row)
        try:
            parsed.append({
                "id": row["id"],
                "timestamp": _parse_timestamp(row["timestamp"])
            })
        except Exception:
            continue

    parsed.sort(key=lambda x: x["timestamp"])
    alerts = []

    for left, right in find_sliding_windows(parsed, window, threshold):
        print("left, right = "+str(left)+","+str(right))
        alert = {
            "rule_id": rule["id"],
            "triggered_at": parsed[right]["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
            "message": f"Message '{message}' repeated {right - left + 1} times in {rule['window_minutes']}m",
            "related_log_ids": [e["id"] for e in parsed[left:right + 1]]
        }
        alerts.append(alert)
        break

    return alerts


def _inactivity_alerts(cur, rule):
    """
    Detects alerts when no logs from a specific service have been recorded within
    a maximum allowed idle time window.

    :param cur: SQLite database cursor
    :param rule: Rule dictionary containing keys:
                 'service', 'max_idle_minutes', 'id'
    :return: List of triggered alert dictionaries
    """
    service = rule["service"]
    max_idle = timedelta(minutes=rule["max_idle_minutes"])

    cur.execute("SELECT MAX(timestamp) as latest FROM logs WHERE service = ?", (service,))
    row = cur.fetchone()
    alerts = []
    if row and row["latest"]:
        last_ts = _parse_timestamp(row["latest"])
        now = datetime.now(timezone.utc)
        if now - last_ts > max_idle:
            alert = {
                "rule_id": rule["id"],
                "triggered_at": now.strftime("%Y-%m-%d %H:%M:%S"),
                "message": f"No logs from '{service}' in the past {rule['max_idle_minutes']} minutes",
                "related_log_ids": []
            }
            alerts.append(alert)
    return alerts


def evaluate_rules(db_path: str, rules_path: str) -> list[dict]:
    """
    Applies all detection rules on the logs stored in the database and returns triggered alerts.

    :param db_path: Path to the SQLite database file
    :param rules_path: Path to the JSON rules configuration file
    :return: List of alert dictionaries triggered by the rules
    """
    print("in evaluate_rules")
    with open(rules_path, 'r') as rules_file:
        rules = json.load(rules_file)

    con = get_db_connection(db_path)
    cur = con.cursor()
    triggered_alerts = []

    for rule in rules:
        rule_type = rule.get("rule_type")
        if rule_type == "keyword_threshold":
            triggered_alerts.extend(_keyword_threshold_alerts(cur, rule))
        elif rule_type == "repeated_message":
            print("repeated message rule")
            triggered_alerts.extend(_repeated_message_alerts(cur, rule))
        elif rule_type == "inactivity":
            triggered_alerts.extend(_inactivity_alerts(cur, rule))
        else:
            # Unknown rule_type: ignore or log if needed
            continue

    con.close()
    return triggered_alerts




# import json
# from datetime import datetime, timedelta
# from db.init_db import get_db_connection

# def evaluate_rules(db_path: str, rules_path: str) -> list[dict]:
#     """
#     Applies all detection rules on the logs in the database.

#     :param db_path: Path to the SQLite database file
#     :param rules_path: Path to the rules_config.json file
#     :return: List of alerts triggered (as dictionaries)
#     """
#     with open(rules_path, 'r') as rules_file:
#         rules = json.load(rules_file)

#     con = get_db_connection(db_path)
#     cur = con.cursor()
#     triggered_alerts = []

#     for rule in rules:
#         rule_type = rule["rule_type"]
#         if rule_type == "keyword_threshold":
#             ...
        
#             service = rule["service"]
#             keyword = rule["keyword"]
#             threshold = rule["threshold"]
#             window_minutes = rule["window_minutes"]

#             # Safely query logs matching service and keyword
#             cur.execute(
#                 "SELECT id, timestamp, message FROM logs WHERE service = ? AND message LIKE ?",
#                 (service, f"%{keyword}%")
#             )
#             entries = cur.fetchall()

#             parsed = []
#             for row in entries:
#                 try:
#                     ts = datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M:%S")
#                     parsed.append({
#                         "id": row["id"],
#                         "timestamp": ts,
#                         "message": row["message"]
#                     })
#                 except Exception:
#                     continue

#             parsed.sort(key=lambda x: x["timestamp"])

#             # Two-pointer sliding window
#             left = 0
#             for right in range(len(parsed)):
#                 while parsed[right]["timestamp"] - parsed[left]["timestamp"] > timedelta(minutes=window_minutes):
#                     left += 1
#                 window_size = right - left + 1
#                 if window_size >= threshold:
#                     window_slice = parsed[left:right + 1]
#                     alert = {
#                         "rule_id": rule["id"],
#                         "triggered_at": parsed[right]["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
#                         "message": f"Rule {rule['id']} triggered with {window_size} matches",
#                         "related_log_ids": [e["id"] for e in window_slice]
#                     }
#                     triggered_alerts.append(alert)
#                     break  # Only one alert per rule

#     con.close()
#     return triggered_alerts
