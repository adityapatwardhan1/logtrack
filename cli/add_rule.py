# This is a dev hack not something for production

import argparse
import getpass
import sqlite3
import hashlib
from auth.auth import verify_user
from db.init_db import get_db_connection


def prompt_login():
    username = input("Username: ")
    password = getpass.getpass("Password: ")
    return username, password


def is_admin_user(db_path, username, password):
    con = get_db_connection(db_path)
    cur = con.cursor()
    cur.execute("SELECT password_hash, role FROM users WHERE username = ?", (username,))
    row = cur.fetchone()
    con.close()

    if row is None:
        return False

    password_hash, role = row
    return verify_user(password, password_hash) and role == "admin"


def add_rule_to_db(db_path, rule):
    con = get_db_connection(db_path)
    cur = con.cursor()
    cur.execute("""
        INSERT OR REPLACE INTO rules (
            id, rule_type, service, keyword, message,
            threshold, window_minutes, max_idle_minutes,
            user_field, description, created_by, window_seconds
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        rule.get("id"),
        rule.get("rule_type"),
        rule.get("service"),
        rule.get("keyword"),
        rule.get("message"),
        rule.get("threshold"),
        rule.get("window_minutes"),
        rule.get("max_idle_minutes"),
        rule.get("user_field"),
        rule.get("description", ""),
        rule.get("created_by"),  # should be set after login
        rule.get("window_seconds"),
    ))
    con.commit()
    con.close()


def main():
    parser = argparse.ArgumentParser(description="Add a detection rule (admin-only).")
    parser.add_argument("--db", required=True, help="Path to the SQLite database.")
    parser.add_argument("--rule-file", required=True, help="Path to JSON file with rule definition.")
    args = parser.parse_args()

    import json
    with open(args.rule_file) as f:
        rule = json.load(f)

    username, password = prompt_login()

    if not is_admin_user(args.db, username, password):
        print("Authentication failed or insufficient privileges. Admin access required.")
        return

    rule["created_by"] = username  # assuming you track username not user ID — can be adjusted
    add_rule_to_db(args.db, rule)
    print(f"Rule '{rule['id']}' added by admin '{username}'.")


if __name__ == "__main__":
    main()