import unittest
from core.parser import parse_log_line

class TestParseLogLine(unittest.TestCase):

    def test_basic_log_with_user(self):
        line = "2025-07-11 13:22:45 auth: login failed for user alice"
        expected = {
            "timestamp": "2025-07-11 13:22:45",
            "service": "auth",
            "message": "login failed for user alice",
            "user": "alice"
        }
        self.assertEqual(parse_log_line(line), expected)

    def test_log_with_colon_in_message(self):
        line = "2025-07-11 14:00:00 system: warning: disk usage at 90%"
        expected = {
            "timestamp": "2025-07-11 14:00:00",
            "service": "system",
            "message": "warning: disk usage at 90%",
            "user": None
        }
        self.assertEqual(parse_log_line(line), expected)

    def test_log_without_user(self):
        line = "2025-07-11 15:00:00 network: ping timeout"
        expected = {
            "timestamp": "2025-07-11 15:00:00",
            "service": "network",
            "message": "ping timeout",
            "user": None
        }
        self.assertEqual(parse_log_line(line), expected)

    def test_log_missing_colon(self):
        line = "2025-07-11 16:00:00 malformed log line"
        expected = {
            "timestamp": "2025-07-11 16:00:00",
            "service": "malformed log line",
            "message": "",
            "user": None
        }
        self.assertEqual(parse_log_line(line), expected)

    def test_log_with_multiple_users(self):
        line = "2025-07-11 17:00:00 auth: user bob impersonated user alice"
        expected = {
            "timestamp": "2025-07-11 17:00:00",
            "service": "auth",
            "message": "user bob impersonated user alice",
            "user": "bob"  # Only first match is returned
        }
        self.assertEqual(parse_log_line(line), expected)

if __name__ == "__main__":
    unittest.main()
