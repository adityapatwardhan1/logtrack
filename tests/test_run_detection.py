import unittest
import os
import sqlite3
import json
from datetime import datetime
from cli.run_detection import main as run_detection_main
import sys

TEST_DB = "tests/test_logtrack.db"
TEST_RULES = "tests/test_rules.json"

def setup_test_db(logs):
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
    cur.executemany("INSERT INTO logs (timestamp, service, message, user) VALUES (?, ?, ?, ?)", logs)
    con.commit()
    con.close()

def write_rules(rules):
    with open(TEST_RULES, "w") as f:
        json.dump(rules, f)

class TestRunDetection(unittest.TestCase):
    def tearDown(self):
        if os.path.exists(TEST_DB):
            os.remove(TEST_DB)
        if os.path.exists(TEST_RULES):
            os.remove(TEST_RULES)

    def test_alert_inserted_to_db(self):
        logs = [
            ("2025-07-11 13:00:00", "auth", "login failed for user alice", "alice"),
            ("2025-07-11 13:01:00", "auth", "login failed for user bob", "bob"),
        ]
        rules = [{
            "id": "auth_fail",
            "service": "auth",
            "keyword": "login failed",
            "threshold": 2,
            "window_minutes": 10
        }]

        setup_test_db(logs)
        write_rules(rules)

        sys.argv = ["cli/run_detection.py", "--db-path", TEST_DB, "--rules-path", TEST_RULES]
        run_detection_main()

        # Verify alert was inserted
        con = sqlite3.connect(TEST_DB)
        cur = con.cursor()
        cur.execute("SELECT * FROM alerts")
        alerts = cur.fetchall()
        con.close()
        self.assertEqual(len(alerts), 1)
        self.assertIn("auth_fail", alerts[0][1])  # rule_id

    def test_invalid_db(self):
        sys.argv = ["cli/run_detection.py", "--db-path", "fake.db", "--rules-path", TEST_RULES]
        write_rules([])

        try:
            run_detection_main()
        except SystemExit:
            pass  # Expected exit on OperationalError

    def test_no_trigger(self):
        logs = [
            ("2025-07-11 13:00:00", "auth", "login failed for user alice", "alice"),
            ("2025-07-11 13:15:00", "auth", "login failed for user bob", "bob"),
        ]
        rules = [{
            "id": "auth_fail_tight_window",
            "service": "auth",
            "keyword": "login failed",
            "threshold": 2,
            "window_minutes": 5
        }]
        setup_test_db(logs)
        write_rules(rules)

        sys.argv = ["cli/run_detection.py", "--db-path", TEST_DB, "--rules-path", TEST_RULES]
        run_detection_main()

        con = sqlite3.connect(TEST_DB)
        cur = con.cursor()
        cur.execute("SELECT * FROM alerts")
        alerts = cur.fetchall()
        con.close()
        self.assertEqual(len(alerts), 0)

if __name__ == "__main__":
    unittest.main()
