import re

def parse_log_line(log_line: str):
    """
    Parses a raw log line into a structured dictionary of fields.

    :param log_line: A single line from the log file
    :type log_line: str

    :returns: A dictionary with extracted fields such as timestamp, service, message, and user (if found)
    :rtype: dict
    """
    # Parse <TIMESTAMP> <SERVICE>: <MESSAGE>
    # Assume format YYYY-MM-DD HH:MM:SS service: message
    timestamp_len = 19  # len("YYYY-MM-DD HH:MM:SS")
    timestamp = log_line[:timestamp_len]
    rest = log_line[timestamp_len:].strip()

    if not ":" in rest:
        return {
            "timestamp": timestamp,
            "service": rest,
            "message": "",
            "user": "unknown"
        }

    first_colon_index = rest.index(":")
    service = rest[:first_colon_index]
    message = rest[first_colon_index + 1:].strip()

    # Extract user (if any, e.g. user alice)
    user_match = re.search(r"\buser\s+([a-zA-Z0-9_]+)", message)
    user = user_match.group(1) if user_match else None

    return {
        "timestamp": timestamp,
        "service": service,
        "message": message,
        "user": user
    }
