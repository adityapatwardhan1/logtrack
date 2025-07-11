import unittest
import os
import sqlite3
import json
from datetime import datetime
from core.rules_engine import evaluate_rules

TEST_DB = "tests/test_logtrack.db"
TEST_RULES = "tests/test_rules.json"

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
    """)
    cur.executemany("""
        INSERT INTO logs (timestamp, service, message, user)
        VALUES (?, ?, ?, ?)
    """, logs)
    con.commit()
    con.close()

def write_rules(rules):
    with open(TEST_RULES, "w") as f:
        json.dump(rules, f)

class TestRulesEngine(unittest.TestCase):
    
    def tearDown(self):
        if os.path.exists(TEST_DB):
            os.remove(TEST_DB)
        if os.path.exists(TEST_RULES):
            os.remove(TEST_RULES)

    def test_db_file_created(self):
        logs = [
            ("2025-07-11 13:00:00", "auth", "login failed for user alice", "alice"),
        ]
        # Setup DB and insert logs
        setup_test_db(logs)
        # Check if DB file exists
        self.assertTrue(os.path.exists(TEST_DB), f"DB file {TEST_DB} should exist after setup_test_db()")


    def test_triggered_alert(self):
        logs = [
            ("2025-07-11 13:00:00", "auth", "login failed for user alice", "alice"),
            ("2025-07-11 13:05:00", "auth", "login failed for user bob", "bob"),
        ]
        setup_test_db(logs)
        rules = [{
            "id": "auth_fail",
            "service": "auth",
            "keyword": "login failed",
            "threshold": 2,
            "window_minutes": 10
        }]
        write_rules(rules)
        alerts = evaluate_rules(TEST_DB, TEST_RULES)
        self.assertEqual(len(alerts), 1)

    def test_no_alert_triggered(self):
        logs = [
            ("2025-07-11 13:00:00", "auth", "login failed for user alice", "alice"),
            ("2025-07-11 13:20:00", "auth", "login failed for user bob", "bob"),
        ]
        setup_test_db(logs)
        rules = [{
            "id": "auth_fail_slow",
            "service": "auth",
            "keyword": "login failed",
            "threshold": 2,
            "window_minutes": 10
        }]
        write_rules(rules)
        alerts = evaluate_rules(TEST_DB, TEST_RULES)
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
            "service": "auth",
            "keyword": "login failed",
            "threshold": 2,
            "window_minutes": 10
        }]
        write_rules(rules)
        alerts = evaluate_rules(TEST_DB, TEST_RULES)
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
                "service": "auth",
                "keyword": "login failed",
                "threshold": 2,
                "window_minutes": 10
            },
            {
                "id": "db_errors",
                "service": "db",
                "keyword": "connection error",
                "threshold": 3,
                "window_minutes": 5
            }
        ]
        write_rules(rules)
        alerts = evaluate_rules(TEST_DB, TEST_RULES)
        self.assertEqual(len(alerts), 2)
        rule_ids = set(alert["rule_id"] for alert in alerts)
        self.assertIn("auth_fail", rule_ids)
        self.assertIn("db_errors", rule_ids)

if __name__ == "__main__":
    unittest.main()
