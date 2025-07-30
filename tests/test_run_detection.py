# # import unittest
# # import os
# # import sqlite3
# # import json
# # from datetime import datetime
# # from cli.run_detection import main as run_detection_main
# # import sys

# # TEST_DB = "tests/test_logtrack.db"
# # TEST_RULES = "tests/test_rules.json"

# # def setup_test_db(logs):
# #     if os.path.exists(TEST_DB):
# #         os.remove(TEST_DB)
# #     con = sqlite3.connect(TEST_DB)
# #     cur = con.cursor()
# #     cur.executescript("""
# #     CREATE TABLE logs (
# #         id INTEGER PRIMARY KEY,
# #         timestamp TEXT,
# #         service TEXT,
# #         message TEXT,
# #         user TEXT
# #     );
# #     CREATE TABLE alerts (
# #         id INTEGER PRIMARY KEY,
# #         rule_id TEXT,
# #         triggered_at TEXT,
# #         message TEXT,
# #         related_log_ids TEXT
# #     );
# #     """)
# #     cur.executemany("INSERT INTO logs (timestamp, service, message, user) VALUES (?, ?, ?, ?)", logs)
# #     con.commit()
# #     con.close()

# # def write_rules(rules):
# #     with open(TEST_RULES, "w") as f:
# #         json.dump(rules, f)

# # class TestRunDetection(unittest.TestCase):
# #     def tearDown(self):
# #         if os.path.exists(TEST_DB):
# #             os.remove(TEST_DB)
# #         if os.path.exists(TEST_RULES):
# #             os.remove(TEST_RULES)

# #     def test_alert_inserted_to_db(self):
# #         logs = [
# #             ("2025-07-11 13:00:00", "auth", "login failed for user alice", "alice"),
# #             ("2025-07-11 13:01:00", "auth", "login failed for user bob", "bob"),
# #         ]
# #         rules = [{
# #             "id": "auth_fail",
# #             "service": "auth",
# #             "keyword": "login failed",
# #             "threshold": 2,
# #             "window_minutes": 10
# #         }]

# #         setup_test_db(logs)
# #         write_rules(rules)

# #         sys.argv = ["cli/run_detection.py", "--db-path", TEST_DB, "--rules-path", TEST_RULES]
# #         run_detection_main()

# #         # Verify alert was inserted
# #         con = sqlite3.connect(TEST_DB)
# #         cur = con.cursor()
# #         cur.execute("SELECT * FROM alerts")
# #         alerts = cur.fetchall()
# #         con.close()
# #         self.assertEqual(len(alerts), 1)
# #         self.assertIn("auth_fail", alerts[0][1])  # rule_id

# #     def test_invalid_db(self):
# #         sys.argv = ["cli/run_detection.py", "--db-path", "fake.db", "--rules-path", TEST_RULES]
# #         write_rules([])

# #         try:
# #             run_detection_main()
# #         except SystemExit:
# #             pass  # Expected exit on OperationalError

# #     def test_no_trigger(self):
# #         logs = [
# #             ("2025-07-11 13:00:00", "auth", "login failed for user alice", "alice"),
# #             ("2025-07-11 13:15:00", "auth", "login failed for user bob", "bob"),
# #         ]
# #         rules = [{
# #             "id": "auth_fail_tight_window",
# #             "service": "auth",
# #             "keyword": "login failed",
# #             "threshold": 2,
# #             "window_minutes": 5
# #         }]
# #         setup_test_db(logs)
# #         write_rules(rules)

# #         sys.argv = ["cli/run_detection.py", "--db-path", TEST_DB, "--rules-path", TEST_RULES]
# #         run_detection_main()

# #         con = sqlite3.connect(TEST_DB)
# #         cur = con.cursor()
# #         cur.execute("SELECT * FROM alerts")
# #         alerts = cur.fetchall()
# #         con.close()
# #         self.assertEqual(len(alerts), 0)

# # if __name__ == "__main__":
# #     unittest.main()


# import unittest
# import os
# import sqlite3
# from cli.run_detection import main as run_detection_main
# import sys

# TEST_DB = "tests/test_logtrack.db"

# def setup_test_db(logs, rules):
#     if os.path.exists(TEST_DB):
#         os.remove(TEST_DB)
#     con = sqlite3.connect(TEST_DB)
#     cur = con.cursor()
#     cur.executescript("""
#     CREATE TABLE logs (
#         id INTEGER PRIMARY KEY,
#         timestamp TEXT,
#         service TEXT,
#         message TEXT,
#         user TEXT
#     );
#     CREATE TABLE alerts (
#         id INTEGER PRIMARY KEY,
#         rule_id TEXT,
#         triggered_at TEXT,
#         message TEXT,
#         related_log_ids TEXT
#     );
#     CREATE TABLE rules (
#         id TEXT PRIMARY KEY,
#         rule_type TEXT,
#         service TEXT,
#         keyword TEXT,
#         message TEXT,
#         threshold INTEGER,
#         window_minutes INTEGER,
#         window_seconds INTEGER,
#         max_idle_minutes INTEGER
#     );
#     """)
#     cur.executemany("INSERT INTO logs (timestamp, service, message, user) VALUES (?, ?, ?, ?)", logs)
    
#     # Insert rules into rules table
#     for rule in rules:
#         cur.execute("""
#             INSERT INTO rules (id, rule_type, service, keyword, message, threshold, window_minutes, window_seconds, max_idle_minutes)
#             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
#         """, (
#             rule.get("id"),
#             rule.get("rule_type"),
#             rule.get("service"),
#             rule.get("keyword"),
#             rule.get("message"),
#             rule.get("threshold"),
#             rule.get("window_minutes"),
#             rule.get("window_seconds"),
#             rule.get("max_idle_minutes")
#         ))
#     con.commit()
#     con.close()

# class TestRunDetection(unittest.TestCase):
#     def tearDown(self):
#         if os.path.exists(TEST_DB):
#             os.remove(TEST_DB)

#     def test_alert_inserted_to_db(self):
#         logs = [
#             ("2025-07-11 13:00:00", "auth", "login failed for user alice", "alice"),
#             ("2025-07-11 13:01:00", "auth", "login failed for user bob", "bob"),
#         ]
#         rules = [{
#             "id": "auth_fail",
#             "rule_type": "keyword_threshold",
#             "service": "auth",
#             "keyword": "login failed",
#             "threshold": 2,
#             "window_minutes": 10
#         }]

#         setup_test_db(logs, rules)

#         sys.argv = ["run_detection.py", "--db-path", TEST_DB]
#         run_detection_main()

#         con = sqlite3.connect(TEST_DB)
#         cur = con.cursor()
#         cur.execute("SELECT * FROM alerts")
#         alerts = cur.fetchall()
#         con.close()
#         self.assertEqual(len(alerts), 1)
#         self.assertIn("auth_fail", alerts[0][1])  # rule_id

#     def test_invalid_db(self):
#         sys.argv = ["run_detection.py", "--db-path", "fake.db"]
#         try:
#             run_detection_main()
#         except SystemExit:
#             pass  # Expected exit on OperationalError

#     def test_no_trigger(self):
#         logs = [
#             ("2025-07-11 13:00:00", "auth", "login failed for user alice", "alice"),
#             ("2025-07-11 13:15:00", "auth", "login failed for user bob", "bob"),
#         ]
#         rules = [{
#             "id": "auth_fail_tight_window",
#             "rule_type": "keyword_threshold",
#             "service": "auth",
#             "keyword": "login failed",
#             "threshold": 2,
#             "window_minutes": 5
#         }]
#         setup_test_db(logs, rules)

#         sys.argv = ["run_detection.py", "--db-path", TEST_DB]
#         run_detection_main()

#         con = sqlite3.connect(TEST_DB)
#         cur = con.cursor()
#         cur.execute("SELECT * FROM alerts")
#         alerts = cur.fetchall()
#         con.close()
#         self.assertEqual(len(alerts), 0)

# if __name__ == "__main__":
#     unittest.main()

import unittest
import os
import sqlite3
from cli import ingest_logs, run_detection
import tempfile
import sys

TEST_DB = "tests/test_logtrack_multi.db"

class TestMultiLogIngestionAndDetection(unittest.TestCase):

    def setUp(self):
        # Remove test DB if exists
        if os.path.exists(TEST_DB):
            os.remove(TEST_DB)

        # Setup empty DB with schema + rules table and insert rules
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
        # Insert a keyword_threshold rule for "login failed" in auth
        cur.execute("""
        INSERT INTO rules (id, rule_type, service, keyword, message, threshold, window_minutes)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, ("auth_fail", "keyword_threshold", "auth", "login failed", None, 3, 10))
        con.commit()
        con.close()

    def tearDown(self):
        if os.path.exists(TEST_DB):
            os.remove(TEST_DB)

    def _write_file(self, filename, contents):
        with open(filename, "w") as f:
            f.write(contents)

    def test_log_ingestion_and_detection(self):
        # Create a temp directory for log files
        with tempfile.TemporaryDirectory() as tmpdir:
            # Regular .log file (uses your parse_log_line)
            log_path = os.path.join(tmpdir, "sample.log")
            log_content = (
                "2025-07-11 13:00:00 auth: login failed for user alice\n"
                "2025-07-11 13:01:00 auth: login failed for user bob\n"
                "2025-07-11 13:02:00 system: disk warning\n"
            )
            self._write_file(log_path, log_content)

            # JSON file, list of dicts
            json_path = os.path.join(tmpdir, "sample.json")
            json_content = """
            [
                {"timestamp": "2025-07-11T13:00:00", "service": "auth", "message": "login failed for user alice", "user": "alice"},
                {"timestamp": "2025-07-11T13:01:00", "service": "auth", "message": "login failed for user bob", "user": "bob"},
                {"timestamp": "2025-07-11T13:02:00", "service": "system", "message": "disk warning"}
            ]
            """
            self._write_file(json_path, json_content)

            # CSV file (HDFS format)
            csv_path = os.path.join(tmpdir, "sample.csv")
            csv_content = (
                "Date,Time,Component,Content,User\n"
                "250711,130000,auth,login failed for user alice,alice\n"
                "250711,130100,auth,login failed for user bob,bob\n"
                "250711,130200,system,disk warning,\n"
            )
            self._write_file(csv_path, csv_content)

            for file_path in [log_path, json_path, csv_path]:
                # Clear DB logs and alerts before each ingestion test
                con = sqlite3.connect(TEST_DB)
                cur = con.cursor()
                cur.execute("DELETE FROM logs")
                cur.execute("DELETE FROM alerts")
                con.commit()
                con.close()

                # Ingest logs from the file
                ingest_logs.ingest_log_file(file_path, db_path=TEST_DB)

                # Run detection on DB
                sys_argv_backup = sys.argv
                sys.argv = ["run_detection.py", "--db-path", TEST_DB]
                try:
                    run_detection.main()
                finally:
                    sys.argv = sys_argv_backup

                # Verify alerts inserted due to keyword_threshold rule on "login failed"
                con = sqlite3.connect(TEST_DB)
                cur = con.cursor()
                cur.execute("SELECT * FROM alerts")
                alerts = cur.fetchall()
                con.close()

                self.assertGreaterEqual(len(alerts), 1)
                # Check alert message contains "login failed"
                self.assertTrue(any("login failed" in alert[3] for alert in alerts))

if __name__ == "__main__":
    unittest.main()
