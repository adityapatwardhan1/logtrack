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

CREATE TABLE IF NOT EXISTS rules (
    id TEXT PRIMARY KEY,
    rule_type TEXT NOT NULL,
    service TEXT,
    keyword TEXT,
    message TEXT,
    threshold INTEGER,
    window_minutes INTEGER,
    window_seconds INTEGER,
    max_idle_minutes INTEGER,
    user_field TEXT,
    description TEXT,
    created_by INTEGER,
    FOREIGN KEY (created_by) REFERENCES users(id)
);