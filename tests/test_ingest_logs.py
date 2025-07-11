import unittest
from unittest.mock import patch, mock_open, MagicMock
from cli import ingest_logs

class TestIngestLogs(unittest.TestCase):

    @patch('ingest_logs.os.path.exists', return_value=True)
    @patch('builtins.open', new_callable=mock_open, read_data="2025-07-11 13:22:45 auth: login failed for user alice\n2025-07-11 14:00:00 system: warning: disk usage at 90%\n")
    @patch('ingest_logs.get_db_connection')
    def test_ingest_log_file_success(self, mock_get_db, mock_file, mock_exists):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_get_db.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor

        ingest_logs.ingest_log_file('dummy.log')

        # Verify file existence was checked
        mock_exists.assert_called_once_with('dummy.log')
        # Verify file opened for reading
        mock_file.assert_called_once_with('dummy.log', 'r')

        # Verify cursor.executemany was called with expected data
        expected_data = [
            ['2025-07-11 13:22:45', 'auth', 'login failed for user alice', 'alice'],
            ['2025-07-11 14:00:00', 'system', 'warning: disk usage at 90%', None]
        ]
        mock_cursor.executemany.assert_called_once_with(
            'INSERT INTO logs (timestamp, service, message, user) VALUES (?, ?, ?, ?)',
            expected_data
        )

        # Verify commit and close called
        mock_conn.commit.assert_called_once()
        mock_conn.close.assert_called_once()

    @patch('ingest_logs.os.path.exists', return_value=False)
    def test_file_not_found(self, mock_exists):
        with self.assertRaises(SystemExit):
            ingest_logs.ingest_log_file('missing.log')
        mock_exists.assert_called_once_with('missing.log')


if __name__ == '__main__':
    unittest.main()
