import streamlit as st
import sqlite3
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from db.init_db import get_db_connection

ph = PasswordHasher()

def register_user(username: str, password: str, role: str = "user"):
    try:
        password_hash = ph.hash(password)
    except Exception as e:
        print(f"Error hashing password: {e}")
        return

    con = get_db_connection()
    cur = con.cursor()
    try:
        cur.execute(
            "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
            (username, password_hash, role)
        )
        con.commit()
        print(f"User '{username}' added with role '{role}'")
    except sqlite3.IntegrityError:
        print(f"Username '{username}' already exists.")
    except Exception as e:
        print(f"Error inserting user '{username}': {e}")
    finally:
        con.close()


def verify_user(username: str, input_password: str) -> bool:
    """Determines whether a username-password combination exists"""

    con = get_db_connection()
    cur = con.cursor()
    cur.execute("SELECT password_hash FROM users WHERE username = ?", (username,))
    row = cur.fetchone()
    con.close()

    if not row:
        return False

    try:
        ph.verify(row["password_hash"], input_password)
        return True
    except VerifyMismatchError:
        return False
