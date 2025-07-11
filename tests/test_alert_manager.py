import unittest
import sqlite3
import os
from db.init_db import init_db
from core.alert_manager import record_alert
from db.init_db import get_db_connection

class TestAlertManager(unittest.TestCase):

    def setUp(self):
        # Use a temporary test database
        self.test_db = "test_logtrack.db"
        if os.path.exists(self.test_db):
            os.remove(self.test_db)
        init_db(self.test_db)

    def tearDown(self):
        if os.path.exists(self.test_db):
            os.remove(self.test_db)

    def test_record_alert(self):
        alert = {
            "rule_id": "rule-test-1",
            "triggered_at": "2025-07-11 12:00:00",
            "message": "Test alert triggered",
            "related_log_ids": [1, 2, 3]
        }

        record_alert(alert, db_path=self.test_db)

        # Check DB for inserted alert
        con = get_db_connection(self.test_db)
        cur = con.cursor()
        cur.execute("SELECT rule_id, triggered_at, message, related_log_ids FROM alerts")
        result = cur.fetchone()

        self.assertIsNotNone(result)
        self.assertEqual(result["rule_id"], alert["rule_id"])
        self.assertEqual(result["triggered_at"], alert["triggered_at"])
        self.assertEqual(result["message"], alert["message"])
        self.assertEqual(result["related_log_ids"], "1,2,3")

        con.close()

if __name__ == '__main__':
    unittest.main()
