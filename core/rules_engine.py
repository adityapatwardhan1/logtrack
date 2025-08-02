import math
from datetime import datetime, timedelta, timezone
from collections import defaultdict
from db.init_db import get_db_connection
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os


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
# Oftentimes it's better to use keyword threshold
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
    threshold = rule["threshold"]
    window = timedelta(minutes=rule["window_minutes"])

    cur.execute(
        "SELECT id, timestamp FROM logs WHERE message = ?", (message,)
    )
    rows = cur.fetchall()

    parsed = []
    for row in rows:
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


def _rate_spike_alerts(cur, rule):
    """
    Detects if a service generates more than `threshold` log entries within `window_seconds`.

    :param cur: SQLite cursor
    :param rule: Rule dictionary with 'service', 'threshold', 'window_seconds', 'id'
    :return: List of triggered alerts
    """
    service = rule["service"]
    threshold = rule["threshold"]
    window_seconds = rule.get("window_seconds")

    if window_seconds is None:
        window_minutes = rule.get("window_minutes")
        if window_minutes is None:
            raise ValueError("Rate spike rule must have either 'window_seconds' or 'window_minutes'")
        window_seconds = window_minutes * 60

    window = timedelta(seconds=window_seconds)

    cur.execute(
        "SELECT id, timestamp FROM logs WHERE service = ?",
        (service,)
    )
    rows = cur.fetchall()

    parsed = []
    for row in rows:
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
        alert = {
            "rule_id": rule["id"],
            "triggered_at": parsed[right]["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
            "message": f"{service} logged {right - left + 1} entries in {rule['window_seconds']}s",
            "related_log_ids": [e["id"] for e in parsed[left:right + 1]]
        }
        alerts.append(alert)
        break

    return alerts


def _user_threshold_alerts(cur, rule):
    """
    Detects alerts where a specific user triggers a given message more than N times in a window.

    :param cur: SQLite cursor
    :param rule: Rule dict with 'message', 'threshold', 'window_minutes', 'id'
    :return: List of alerts triggered
    """
    message = rule["message"]
    threshold = rule["threshold"]
    window = timedelta(minutes=rule["window_minutes"])

    cur.execute("SELECT id, timestamp, user FROM logs WHERE message LIKE ? AND user IS NOT NULL", (f"%{message}%",))
    rows = cur.fetchall()

    # Group by user
    events_by_user = {}
    for row in rows:
        if row["user"] not in events_by_user:
            events_by_user[row["user"]] = []
        try:
            events_by_user[row["user"]].append({
                "id": row["id"],
                "timestamp": _parse_timestamp(row["timestamp"])
            })
        except Exception:
            continue

    alerts = []

    for user, events in events_by_user.items():
        events.sort(key=lambda e: e["timestamp"])
        for left, right in find_sliding_windows(events, window, threshold):
            alert = {
                "rule_id": rule["id"],
                "triggered_at": events[right]["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
                "message": f"User '{user}' triggered '{message}' {right - left + 1} times in {rule['window_minutes']}m",
                "related_log_ids": [e["id"] for e in events[left:right + 1]]
            }
            alerts.append(alert)
            break  # One alert per user
    return alerts


def _zscore_alerts(cur, rule):
    """
    Detects log volume spikes for a service using z-score anomaly detection.

    :param cur: SQLite cursor
    :param rule: Rule dict with 'service', 'threshold', 'window_minutes', 'baseline_windows', 'id'
    :return: List of alerts triggered
    """
    service = rule.get("service")
    threshold = rule.get("zscore_threshold") or rule.get("threshold") or 3  # Default threshold
    window_minutes = rule.get("window_minutes", 5)
    baseline_windows = rule.get("baseline_windows", 6)

    if not service or threshold is None:
        print("Missing service or threshold")
        return []

    # Fetch all logs for the service
    cur.execute("""
        SELECT id, timestamp FROM logs
        WHERE service = ?
          AND timestamp IS NOT NULL
        ORDER BY timestamp ASC
    """, (service,))
    rows = cur.fetchall()

    if not rows:
        print("No logs found")
        return []

    # Parse timestamps and find latest (reference "now")
    log_times = []
    for log_id, ts in rows:
        try:
            dt = _parse_timestamp(ts)
            log_times.append((log_id, dt))
        except Exception:
            continue

    if not log_times:
        print("No valid timestamps")
        return []

    # Use latest log time as "now"
    max_time = max(dt for _, dt in log_times)
    now = max_time

    # Bin logs into windows: bucket 0 = oldest, bucket N = most recent
    buckets = defaultdict(list)
    for log_id, dt in log_times:
        delta_minutes = (now - dt).total_seconds() / 60
        bucket_idx = baseline_windows - int(delta_minutes // window_minutes)
        if 0 <= bucket_idx <= baseline_windows:
            buckets[bucket_idx].append((log_id, dt))

    if len(buckets) < baseline_windows + 1:
        return []

    # Sort buckets by index: oldest → newest
    sorted_indices = list(range(baseline_windows + 1))
    baseline_counts = [len(buckets[idx]) for idx in sorted_indices[:-1]]
    current_count = len(buckets[sorted_indices[-1]])
    current_bucket_log_ids = [log_id for log_id, _ in buckets[sorted_indices[-1]]]

    # Compute z-score
    mean = sum(baseline_counts) / len(baseline_counts)
    std = math.sqrt(sum((x - mean) ** 2 for x in baseline_counts) / len(baseline_counts)) if baseline_counts else 0
    alerts = []
    if std > 0:
        z = (current_count - mean) / std
        if z >= threshold:
            alerts.append({
                "rule_id": rule["id"],
                "triggered_at": now.isoformat(),
                "message": f"{service} log volume spike detected (z={z:.2f})",
                "related_log_ids": current_bucket_log_ids
            })

    elif std == 0 and current_count > mean * threshold:
        alerts.append({
            "rule_id": rule["id"],
            "triggered_at": now.isoformat(),
            "message": f"{service} log volume spike detected (std=0, count={current_count}, mean={mean})",
            "related_log_ids": current_bucket_log_ids
        })

    return alerts


def run_ml_detection(con, model_path, threshold_path, feature_extractor_paths):
    """
    Run trained ML model on logs fetched from DB connection `con`.
    feature_extractor_paths is a dict with keys:
      - 'tfidf', 'scaler', 'service_encoder', 'user_hasher'
    """

    import pandas as pd
    import numpy as np
    import os
    import joblib
    from scipy.sparse import hstack
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_extraction.text import TfidfVectorizer, FeatureHasher
    from sklearn.preprocessing import OneHotEncoder

    print("In run_ml_detection...")

    # Load logs from DB ordered by timestamp
    df_logs = pd.read_sql_query("SELECT timestamp, service, message, user FROM logs ORDER BY timestamp", con)
    con.close()

    # === Derive features ===
    df_logs['timestamp'] = pd.to_datetime(df_logs['timestamp'])

    # Extract BlockId from message
    df_logs['BlockId'] = df_logs['message'].str.extract(r'(blk_-?\d+)')

    # Load saved block count mapping
    block_count_path = feature_extractor_paths['block_count']
    if not os.path.exists(block_count_path):
        raise FileNotFoundError(f"Missing block count mapping at {block_count_path}")
    block_counts = joblib.load(block_count_path)

    # Apply consistent mapping
    df_logs['block_count'] = df_logs['BlockId'].map(block_counts).fillna(0).astype(int)

    # Timestamp features
    df_logs['hour'] = df_logs['timestamp'].dt.hour
    df_logs['dayofweek'] = df_logs['timestamp'].dt.dayofweek

    # Load feature transformers
    tfidf = joblib.load(feature_extractor_paths['tfidf'])
    scaler = joblib.load(feature_extractor_paths['scaler'])
    service_enc = joblib.load(feature_extractor_paths['service_encoder'])
    hasher = joblib.load(feature_extractor_paths['user_hasher'])

    # Transform features
    X_text = tfidf.transform(df_logs['message'])
    X_service = service_enc.transform(df_logs[['service']])
    user_data = df_logs['user'].fillna("unknown").astype(str).apply(lambda x: [x]).tolist()
    X_user = hasher.transform(user_data)

    # Numeric features to scale
    X_numeric = df_logs[['hour', 'dayofweek', 'block_count']].astype(np.float32).values
    X_numeric_scaled = scaler.transform(X_numeric)

    # Combine all features
    X_combined = hstack([X_text, X_numeric_scaled, X_service, X_user])

    # Load model
    model = joblib.load(model_path)

    # Load threshold
    with open(threshold_path, 'r') as threshold_file:
        threshold = int(threshold_file.readline().strip('\n'))

    # Predict
    y_proba = model.predict_proba(X_combined)
    y_pred = (y_proba[:, 1] > threshold).astype(int)

    # Format alerts
    alerts = []
    for i, is_anomaly in enumerate(y_pred):
        if is_anomaly == 1:
            ts, svc, msg = df_logs.iloc[i][['timestamp', 'service', 'message']]
            alerts.append({
                "rule_id": "ml_anomaly",
                "triggered_at": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "message": f"[ML] Anomaly detected by ML on service '{svc}' — '{msg}'",
                "severity": 2,
                "related_log_ids": [],
                "data": {}
            })

    print(f"[ML] {len(alerts)} anomalies detected")
    return alerts


def evaluate_rules(db_path: str, zscore_enabled=False, ml_enabled=False) -> list[dict]:
    """
    Applies all detection rules on the logs stored in the database and returns triggered alerts.

    :param db_path: Path to the SQLite database file
    :param zscore_enabled: Whether to evaluate z-score rules
    :param ml_enabled: Whether to evaluate ML-based anomaly detection
    :return: List of alert dictionaries triggered by the rules
    """
    con = get_db_connection(db_path)
    cur = con.cursor()

    cur.execute("SELECT * FROM rules")
    rules = [dict(row) for row in cur.fetchall()]
    triggered_alerts = []

    for rule in rules:
        rule_type = rule.get("rule_type")
        if rule_type == "keyword_threshold":
            triggered_alerts.extend(_keyword_threshold_alerts(cur, rule))
        elif rule_type == "repeated_message":
            triggered_alerts.extend(_repeated_message_alerts(cur, rule))
        elif rule_type == "inactivity":
            triggered_alerts.extend(_inactivity_alerts(cur, rule))
        elif rule_type == "rate_spike":
            triggered_alerts.extend(_rate_spike_alerts(cur, rule))
        elif rule_type == "user_threshold":
            triggered_alerts.extend(_user_threshold_alerts(cur, rule))
        elif rule_type == "zscore_anomaly" and zscore_enabled:
            triggered_alerts.extend(_zscore_alerts(cur, rule))
        else:
            if "rule_type" not in rule:
                raise ValueError(f"Missing 'rule_type' in rule: {rule}")
    if ml_enabled:
        feature_extractor_paths = {
            'tfidf': 'saved_feature_extractor/feature_extractor.pkl',
            'scaler': 'saved_feature_extractor/scaler.pkl',
            'service_encoder': 'saved_feature_extractor/service_encoder.pkl',
            'user_hasher': 'saved_feature_extractor/user_hasher.pkl',
            'block_count': 'saved_feature_extractor/block_count_mapping.pkl'
        }
        model_path = 'saved_models/XGBoostClassifier.pkl'
        threshold_path = 'saved_models/threshold.txt'
        triggered_alerts.extend(run_ml_detection(con, model_path, threshold_path, feature_extractor_paths))

    con.close()
    return triggered_alerts
