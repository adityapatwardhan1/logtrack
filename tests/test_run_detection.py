import unittest
import os
import sqlite3
import tempfile
import sys

from cli import ingest_logs, run_detection

TEST_DB = "tests/test_logtrack_multi.db"

class TestMultiLogIngestionAndDetection(unittest.TestCase):

    def setUp(self):
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
            CREATE TABLE rules (
                id TEXT PRIMARY KEY,
                rule_type TEXT,
                service TEXT,
                keyword TEXT,
                message TEXT,
                threshold INTEGER,
                window_minutes INTEGER,
                window_seconds INTEGER,
                max_idle_minutes INTEGER
            );
        """)

        # Add threshold rule for "login failed" in auth service
        cur.execute("""
            INSERT INTO rules (id, rule_type, service, keyword, message, threshold, window_minutes)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, ("auth_fail", "keyword_threshold", "auth", "login failed", None, 1, 10))

        con.commit()
        con.close()

    def tearDown(self):
        if os.path.exists(TEST_DB):
            os.remove(TEST_DB)

    def _write_file(self, path, content):
        with open(path, "w") as f:
            f.write(content)

    def test_log_ingestion_and_detection(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # --- Sample log file (.log)
            log_path = os.path.join(tmpdir, "sample.log")
            self._write_file(log_path, (
                "2025-07-11 13:00:00 auth: login failed for user alice\n"
                "2025-07-11 13:01:00 auth: login failed for user bob\n"
                "2025-07-11 13:02:00 system: disk warning\n"
            ))

            # --- Sample JSON log file
            json_path = os.path.join(tmpdir, "sample.json")
            self._write_file(json_path, """
                [
                    {"timestamp": "2025-07-11T13:00:00", "service": "auth", "message": "login failed for user alice", "user": "alice"},
                    {"timestamp": "2025-07-11T13:01:00", "service": "auth", "message": "login failed for user bob", "user": "bob"},
                    {"timestamp": "2025-07-11T13:02:00", "service": "system", "message": "disk warning"}
                ]
            """)

            # --- Sample HDFS-style CSV
            csv_path = os.path.join(tmpdir, "sample.csv")
            self._write_file(csv_path, (
                "Date,Time,Component,Content,User\n"
                "250711,130000,auth,login failed for user alice,alice\n"
                "250711,130100,auth,login failed for user bob,bob\n"
                "250711,130200,system,disk warning,\n"
            ))

            # Test all log types
            for file_path in [log_path, json_path, csv_path]:
                self._reset_db_logs_and_alerts()

                ingest_logs.ingest_log_file(file_path, db_path=TEST_DB)

                # Simulate CLI call to run_detection
                sys_argv_backup = sys.argv
                sys.argv = ["run_detection.py", "--db-path", TEST_DB]
                try:
                    run_detection.main()
                finally:
                    sys.argv = sys_argv_backup

                alerts = self._fetch_alerts()
                self.assertGreaterEqual(len(alerts), 1)
                self.assertTrue(any("auth_fail" in alert[1] for alert in alerts))

    def _reset_db_logs_and_alerts(self):
        con = sqlite3.connect(TEST_DB)
        cur = con.cursor()
        cur.execute("DELETE FROM logs")
        cur.execute("DELETE FROM alerts")
        con.commit()
        con.close()

    def _fetch_alerts(self):
        con = sqlite3.connect(TEST_DB)
        cur = con.cursor()
        cur.execute("SELECT * FROM alerts")
        results = cur.fetchall()
        con.close()
        return results


if __name__ == "__main__":
    unittest.main()
