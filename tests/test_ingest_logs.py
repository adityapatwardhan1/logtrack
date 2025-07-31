# import unittest
# from unittest.mock import patch, mock_open, MagicMock
# from cli import ingest_logs

# class TestIngestLogs(unittest.TestCase):

#     @patch('cli.ingest_logs.os.path.exists', return_value=True)
#     @patch('builtins.open', new_callable=mock_open, read_data="2025-07-11 13:22:45 auth: login failed for user alice\n2025-07-11 14:00:00 system: warning: disk usage at 90%\n")
#     @patch('cli.ingest_logs.get_db_connection')
#     def test_ingest_log_file_success(self, mock_get_db, mock_file, mock_exists):
#         mock_conn = MagicMock()
#         mock_cursor = MagicMock()
#         mock_get_db.return_value = mock_conn
#         mock_conn.cursor.return_value = mock_cursor

#         ingest_logs.ingest_log_file('dummy.log')

#         # Verify file existence was checked
#         mock_exists.assert_called_once_with('dummy.log')
#         # Verify file opened for reading
#         mock_file.assert_called_once_with('dummy.log', 'r')

#         # Verify cursor.executemany was called with expected data
#         expected_data = [
#             ['2025-07-11 13:22:45', 'auth', 'login failed for user alice', 'alice'],
#             ['2025-07-11 14:00:00', 'system', 'warning: disk usage at 90%', None]
#         ]
#         mock_cursor.executemany.assert_called_once_with(
#             'INSERT INTO logs (timestamp, service, message, user) VALUES (?, ?, ?, ?)',
#             expected_data
#         )

#         # Verify commit and close called
#         mock_conn.commit.assert_called_once()
#         mock_conn.close.assert_called_once()

#     @patch('cli.ingest_logs.os.path.exists', return_value=False)
#     def test_file_not_found(self, mock_exists):
#         with self.assertRaises(SystemExit):
#             ingest_logs.ingest_log_file('missing.log')
#         mock_exists.assert_called_once_with('missing.log')


# if __name__ == '__main__':
#     unittest.main()


import unittest
from unittest.mock import patch, mock_open, MagicMock
from cli import ingest_logs

class TestIngestLogs(unittest.TestCase):

    @patch('cli.ingest_logs.get_db_connection')
    def test_ingest_multiple_formats(self, mock_get_db):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_get_db.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor

        test_cases = [
            {
                "filename": "dummy.log",
                "file_content": "2025-07-11 13:22:45 auth: login failed for user alice\n"
                                "2025-07-11 14:00:00 system: warning: disk usage at 90%\n",
                "expected_data": [
                    ['2025-07-11 13:22:45', 'auth', 'login failed for user alice', 'alice'],
                    ['2025-07-11 14:00:00', 'system', 'warning: disk usage at 90%', None]
                ]
            },
            {
                "filename": "hdfs.csv",
                "file_content": "Date,Time,Pid,Level,Component,Content\n"
                                "01/01/2023,13:00:00,1234,INFO,FS,Successfully fetched block\n",
                "expected_data": [
                    ['2023-01-01 13:00:00', 'FS', 'Successfully fetched block', 'unknown']
                ]
            },
            {
                "filename": "logs.json",
                "file_content": '[{"timestamp": "2023-01-01T13:00:00", "service": "web", '
                                '"message": "user logged in", "user": "bob"}]',
                "expected_data": [
                    ['2023-01-01 13:00:00', 'web', 'user logged in', 'bob']
                ]
            }
        ]

        for case in test_cases:
            with self.subTest(case=case["filename"]):
                with patch('cli.ingest_logs.os.path.exists', return_value=True), \
                     patch('builtins.open', mock_open(read_data=case["file_content"])):

                    ingest_logs.ingest_log_file(case["filename"])

                    mock_cursor.executemany.assert_called_with(
                        'INSERT INTO logs (timestamp, service, message, user) VALUES (?, ?, ?, ?)',
                        case["expected_data"]
                    )
                    mock_conn.commit.assert_called()
                    mock_conn.close.assert_called()
                    mock_cursor.reset_mock()

    @patch('cli.ingest_logs.os.path.exists', return_value=False)
    def test_file_not_found(self, mock_exists):
        with self.assertRaises(SystemExit):
            ingest_logs.ingest_log_file('missing.log')
        mock_exists.assert_called_once_with('missing.log')


if __name__ == '__main__':
    unittest.main()
