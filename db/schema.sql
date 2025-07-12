CREATE TABLE IF NOT EXISTS logs (
    id INTEGER PRIMARY KEY,
    timestamp TEXT,
    service TEXT,
    message TEXT,
    user TEXT
);

CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY,
    rule_id TEXT,
    triggered_at TEXT,
    message TEXT,
    related_log_ids TEXT
);

CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'user'
);

