import unittest
import os
import sqlite3
import json
from datetime import datetime, timedelta, timezone
from core.rules_engine import evaluate_rules

TEST_DB = "tests/test_logtrack.db"

def setup_test_db(logs):
    """Creates test DB, logs schema, and inserts sample logs."""
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)

    con = sqlite3.connect(TEST_DB)
    cur = con.cursor()
    cur.executescript("""
    CREATE TABLE logs (
        id INTEGER PRIMARY KEY,
        timestamp TEXT,
        service TEXT,
        message TEXT,
        user TEXT
    );
    CREATE TABLE alerts (
        id INTEGER PRIMARY KEY,
        rule_id TEXT,
        triggered_at TEXT,
        message TEXT,
        related_log_ids TEXT
    );
    CREATE TABLE IF NOT EXISTS rules (
        id TEXT PRIMARY KEY,
        rule_type TEXT NOT NULL,
        service TEXT,
        keyword TEXT,
        message TEXT,
        threshold INTEGER,
        window_minutes INTEGER,
        window_seconds INTEGER,
        max_idle_minutes INTEGER,
        user_field TEXT,
        description TEXT,
        created_by INTEGER,
        FOREIGN KEY (created_by) REFERENCES users(id)
    );
    """)
    cur.executemany("""
        INSERT INTO logs (timestamp, service, message, user)
        VALUES (?, ?, ?, ?)
    """, logs)
    con.commit()
    con.close()

def write_rules(rules):
    con = sqlite3.connect(TEST_DB)
    cur = con.cursor()
    for rule in rules:
        cur.execute("""
            INSERT OR REPLACE INTO rules (
                id, rule_type, service, keyword, message,
                threshold, window_minutes, window_seconds, max_idle_minutes,
                user_field, description, created_by
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            rule.get("id"),
            rule.get("rule_type"),
            rule.get("service"),
            rule.get("keyword"),
            rule.get("message"),
            rule.get("threshold"),
            rule.get("window_minutes"),
            rule.get("window_seconds"),   # <--- added this
            rule.get("max_idle_minutes"),
            rule.get("user_field"),
            rule.get("description", ""),
            None  # You can assign created_by later, or auto-fill
        ))
    con.commit()
    con.close()


class TestRulesEngine(unittest.TestCase):

    def setUp(self):
        if os.path.exists(TEST_DB):
            os.remove(TEST_DB)

    def tearDown(self):
        if os.path.exists(TEST_DB):
            os.remove(TEST_DB)

    def test_db_file_created(self):
        logs = [
            ("2025-07-11 13:00:00", "auth", "login failed for user alice", "alice"),
        ]
        setup_test_db(logs)
        self.assertTrue(os.path.exists(TEST_DB), f"DB file {TEST_DB} should exist after setup_test_db()")

    def test_triggered_alert(self):
        logs = [
            ("2025-07-11 13:00:00", "auth", "login failed for user alice", "alice"),
            ("2025-07-11 13:05:00", "auth", "login failed for user bob", "bob"),
        ]
        setup_test_db(logs)
        rules = [{
            "id": "auth_fail",
            "rule_type": "keyword_threshold",
            "service": "auth",
            "keyword": "login failed",
            "threshold": 2,
            "window_minutes": 10
        }]
        write_rules(rules)
        alerts = evaluate_rules(TEST_DB)
        self.assertEqual(len(alerts), 1)

    def test_no_alert_triggered(self):
        logs = [
            ("2025-07-11 13:00:00", "auth", "login failed for user alice", "alice"),
            ("2025-07-11 13:20:00", "auth", "login failed for user bob", "bob"),
        ]
        setup_test_db(logs)
        rules = [{
            "id": "auth_fail_slow",
            "rule_type": "keyword_threshold",
            "service": "auth",
            "keyword": "login failed",
            "threshold": 2,
            "window_minutes": 10
        }]
        write_rules(rules)
        alerts = evaluate_rules(TEST_DB)
        self.assertEqual(len(alerts), 0)

    def test_malformed_timestamps(self):
        logs = [
            ("bad timestamp", "auth", "login failed for user alice", "alice"),
            ("2025-07-11 13:05:00", "auth", "login failed for user bob", "bob"),
            ("not a time", "auth", "login failed again", "bob"),
        ]
        setup_test_db(logs)
        rules = [{
            "id": "auth_fail_malformed",
            "rule_type": "keyword_threshold",
            "service": "auth",
            "keyword": "login failed",
            "threshold": 2,
            "window_minutes": 10
        }]
        write_rules(rules)
        alerts = evaluate_rules(TEST_DB)
        self.assertEqual(len(alerts), 0)

    def test_multiple_rules(self):
        logs = [
            ("2025-07-11 13:00:00", "auth", "login failed for user alice", "alice"),
            ("2025-07-11 13:03:00", "auth", "login failed for user bob", "bob"),
            ("2025-07-11 13:10:00", "db", "connection error", None),
            ("2025-07-11 13:11:00", "db", "connection error", None),
            ("2025-07-11 13:12:00", "db", "connection error", None),
        ]
        setup_test_db(logs)
        rules = [
            {
                "id": "auth_fail",
                "rule_type": "keyword_threshold",
                "service": "auth",
                "keyword": "login failed",
                "threshold": 2,
                "window_minutes": 10
            },
            {
                "id": "db_errors",
                "rule_type": "keyword_threshold",
                "service": "db",
                "keyword": "connection error",
                "threshold": 3,
                "window_minutes": 5
            }
        ]
        write_rules(rules)
        alerts = evaluate_rules(TEST_DB)
        self.assertEqual(len(alerts), 2)
        rule_ids = set(alert["rule_id"] for alert in alerts)
        self.assertIn("auth_fail", rule_ids)
        self.assertIn("db_errors", rule_ids)

    def test_rate_spike_alert(self):
        logs = [
            ("2025-07-11 13:00:00", "auth", "event", "alice"),
            ("2025-07-11 13:00:05", "auth", "event", "alice"),
            ("2025-07-11 13:00:10", "auth", "event", "alice"),
            ("2025-07-11 13:00:15", "auth", "event", "alice"),
            ("2025-07-11 13:00:20", "auth", "event", "alice"),
            ("2025-07-11 13:00:25", "auth", "event", "alice"),
            ("2025-07-11 13:00:30", "auth", "event", "alice"),
        ]
        setup_test_db(logs)
        rules = [{
            "id": "auth_spike",
            "rule_type": "rate_spike",
            "service": "auth",
            "threshold": 5,
            "window_seconds": 60
        }]
        write_rules(rules)
        alerts = evaluate_rules(TEST_DB)
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0]["rule_id"], "auth_spike")
        self.assertIn("auth logged", alerts[0]["message"])

    def test_user_threshold_triggered(self):
        logs = [
            ("2025-07-11 13:00:00", "auth", "login failed for user alice", "alice"),
            ("2025-07-11 13:01:30", "auth", "login failed for user alice", "alice"),
            ("2025-07-11 13:03:00", "auth", "login failed for user alice", "alice"),
            ("2025-07-11 13:07:00", "auth", "login failed for user bob", "bob"),  # Outside window
        ]
        setup_test_db(logs)
        rules = [
            {
                "id": "user_login_failures",
                "rule_type": "user_threshold",
                "message": "login failed",
                "threshold": 3,
                "window_minutes": 5
            }
        ]
        write_rules(rules)
        alerts = evaluate_rules(TEST_DB)
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0]["rule_id"], "user_login_failures")
        self.assertIn("alice", alerts[0]["message"])

    def test_user_threshold_not_triggered(self):
        logs = [
            ("2025-07-11 13:00:00", "auth", "login failed for user alice", "alice"),
            ("2025-07-11 13:10:00", "auth", "login failed for user alice", "alice"),
            ("2025-07-11 13:20:00", "auth", "login failed for user alice", "alice"),
        ]
        setup_test_db(logs)
        rules = [
            {
                "id": "user_login_failures",
                "rule_type": "user_threshold",
                "message": "login failed",
                "threshold": 3,
                "window_minutes": 5
            }
        ]
        write_rules(rules)
        alerts = evaluate_rules(TEST_DB)
        self.assertEqual(len(alerts), 0)

    def test_user_threshold_multiple_users(self):
        logs = [
            ("2025-07-11 13:00:00", "auth", "login failed for user alice", "alice"),
            ("2025-07-11 13:01:00", "auth", "login failed for user alice", "alice"),
            ("2025-07-11 13:02:00", "auth", "login failed for user alice", "alice"),
            ("2025-07-11 13:03:00", "auth", "login failed for user bob", "bob"),
            ("2025-07-11 13:04:00", "auth", "login failed for user bob", "bob"),
            ("2025-07-11 13:05:00", "auth", "login failed for user bob", "bob"),
            ("2025-07-11 13:06:00", "auth", "login failed for user bob", "bob"),
        ]
        setup_test_db(logs)
        rules = [
            {
                "id": "user_login_failures",
                "rule_type": "user_threshold",
                "message": "login failed",
                "threshold": 3,
                "window_minutes": 5
            }
        ]
        write_rules(rules)
        alerts = evaluate_rules(TEST_DB)
        self.assertEqual(len(alerts), 2)
        rule_ids = [alert["rule_id"] for alert in alerts]
        self.assertTrue(all(rid == "user_login_failures" for rid in rule_ids))


    def test_zscore_trigger_simple(self):
        logs = [
            # 6 baseline windows with 1 event each exactly on the bucket start time
            ("2025-07-11 11:50:00", "api", "ping", "a"),
            ("2025-07-11 12:00:00", "api", "ping", "a"),
            ("2025-07-11 12:10:00", "api", "ping", "a"),
            ("2025-07-11 12:20:00", "api", "ping", "a"),
            ("2025-07-11 12:30:00", "api", "ping", "a"),
            ("2025-07-11 12:40:00", "api", "ping", "a"),
            # Spike window: 6 events all exactly at bucket start time 12:50:00
            ("2025-07-11 12:50:00", "api", "ping", "a"),
            ("2025-07-11 12:50:00", "api", "ping", "a"),
            ("2025-07-11 12:50:00", "api", "ping", "a"),
            ("2025-07-11 12:50:00", "api", "ping", "a"),
            ("2025-07-11 12:50:00", "api", "ping", "a"),
            ("2025-07-11 12:50:00", "api", "ping", "a"),
        ]
        setup_test_db(logs)
        rules = [{
            "id": "zscore_trigger",
            "rule_type": "zscore_anomaly",
            "service": "api",
            "baseline_windows": 5,
            "window_minutes": 10,
            "threshold": 1.5  # Must use "threshold" here
        }]
        write_rules(rules)
        alerts = evaluate_rules(TEST_DB, zscore_enabled=True)
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0]["rule_id"], "zscore_trigger")
        self.assertIn("log volume spike", alerts[0]["message"])


    def test_zscore_trigger_normal_spike(self):
        """
        Creates 6 baseline windows with varying log counts, and 7 logs in the current window.
        Should trigger alert if zscore_threshold = 2.
        """
        now = datetime(2025, 7, 11, 13, 0, 0, tzinfo=timezone.utc)
        window_minutes = 10
        baseline_windows = 6

        baseline_counts = [1, 3, 2, 1, 4, 2]  # Varying counts to get non-zero std

        logs = []
        for i in range(baseline_windows):
            base_time = now - timedelta(minutes=(baseline_windows - i + 1) * window_minutes)
            for j in range(baseline_counts[i]):
                logs.append((
                    base_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "web",
                    "GET /home",
                    f"user{i}-{j}"
                ))

        # Current window: spike with 7 logs
        current_window_start = now - timedelta(minutes=window_minutes)
        for i in range(7):
            logs.append((
                (current_window_start + timedelta(seconds=i * 30)).strftime("%Y-%m-%d %H:%M:%S"),
                "web",
                "GET /home",
                f"userX-{i}"
            ))

        setup_test_db(logs)
        rules = [{
            "id": "zscore_spike",
            "rule_type": "zscore_anomaly",
            "service": "web",
            "baseline_windows": baseline_windows,
            "window_minutes": window_minutes,
            "threshold": 2.0
        }]
        write_rules(rules)
        alerts = evaluate_rules(TEST_DB, zscore_enabled=True)
        self.assertEqual(len(alerts), 1)
        self.assertIn("log volume spike", alerts[0]["message"])


    def test_zscore_no_alert_normal_distribution(self):
        """
        Creates baseline windows with small random variation around 3 logs per window.
        Current window also has 3 logs — should NOT trigger alert.
        """
        now = datetime(2025, 7, 11, 15, 0, 0, tzinfo=timezone.utc)
        window_minutes = 10
        baseline_windows = 5

        logs = []
        # Baseline: 2-4 logs each window
        counts = [2, 3, 4, 3, 2]
        for i, count in enumerate(counts):
            base_time = now - timedelta(minutes=(baseline_windows - i + 1) * window_minutes)
            for j in range(count):
                logs.append((
                    (base_time + timedelta(seconds=j * 20)).strftime("%Y-%m-%d %H:%M:%S"),
                    "db",
                    "query executed",
                    f"user{i}-{j}"
                ))

        # Current window: same as average
        current_window_start = now - timedelta(minutes=window_minutes)
        for i in range(3):
            logs.append((
                (current_window_start + timedelta(seconds=i * 30)).strftime("%Y-%m-%d %H:%M:%S"),
                "db",
                "query executed",
                f"userX-{i}"
            ))

        setup_test_db(logs)
        rules = [{
            "id": "zscore_no_spike",
            "rule_type": "zscore_anomaly",
            "service": "db",
            "baseline_windows": baseline_windows,
            "window_minutes": window_minutes,
            "threshold": 2.0
        }]
        write_rules(rules)
        alerts = evaluate_rules(TEST_DB, zscore_enabled=True)
        self.assertEqual(len(alerts), 0)



if __name__ == "__main__":
    unittest.main()

