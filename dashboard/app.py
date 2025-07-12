import os 
import sys
import streamlit as st
import sqlite3
import pandas as pd
from pathlib import Path
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from db.init_db import get_db_connection
from dashboard_helpers import verify_user

DB_PATH = Path("logtrack.db")
ph = PasswordHasher()

# -------------------- Auth Functions --------------------
def check_credentials(username, password):
    """
    Determines whether user exists and if so, what their role is.
    :param username: Username of user to check for
    :type username: str
    :param password: Password of user to check for
    :type password: str
    :returns: tuple[bool, Union[str, None]] of whether user exists and role if it does exist
    """
    try:
        con = get_db_connection()
        cur = con.cursor()
        cur.execute("SELECT password_hash, role FROM users WHERE username = ?", (username,))
        row = cur.fetchone()
        con.close()
        if not row:
            return False, None
        stored_hash, role = row
        try:
            ph.verify(stored_hash, password)
            return True, role
        except VerifyMismatchError:
            return False, None
    except Exception as e:
        # Log error internally, but do NOT expose to user
        print(f"Error during credential check for user '{username}': {e}")
        # Optionally, re-raise or handle gracefully
        return False, None

# -------------------- Login Flow --------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title("🔐 LogTrack Login")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")

        if submit:
            try:
                ok, role = check_credentials(username, password)
                if ok:
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.session_state.role = role
                    st.success(f"Welcome, {username}!")
                    st.experimental_rerun()
                else:
                    st.error("Invalid username or password.")
            except Exception:
                # Log unexpected error internally
                print(f"Unexpected error during login attempt for user '{username}':", exc_info=True)
                # Show generic error to user
                st.error("An unexpected error occurred. Please try again later.")
    st.stop()

# -------------------- Main Dashboard --------------------
st.set_page_config(page_title="LogTrack Dashboard", layout="wide")
st.title("📊 LogTrack Monitoring Dashboard")

# Sidebar
st.sidebar.header("🔎 Filters")
log_limit = st.sidebar.slider("Number of logs to show", 10, 500, 100)

def get_connection():
    """Loads database at DB_PATH"""
    if not DB_PATH.exists():
        st.error(f"Database not found at {DB_PATH}. Please run init_db.py first.")
        st.stop()
    return sqlite3.connect(DB_PATH)

def load_logs(limit):
    """Loads up to limit number of logs from database"""
    con = get_connection()
    df = pd.read_sql_query(
        f"SELECT id, timestamp, service, message, user FROM logs ORDER BY timestamp DESC LIMIT {limit}",
        con
    )
    con.close()
    return df

def load_alerts():
    """Loads alerts from database"""
    con = get_connection()
    df = pd.read_sql_query("SELECT * FROM alerts ORDER BY triggered_at DESC", con)
    con.close()
    return df

# Tabs
tabs = st.tabs(["🪵 Logs", "🚨 Alerts"])

# Logs Tab
with tabs[0]:
    st.subheader("Recent Logs")
    logs_df = load_logs(log_limit)
    st.dataframe(logs_df, use_container_width=True)

# Alerts Tab
with tabs[1]:
    st.subheader("Triggered Alerts")
    alerts_df = load_alerts()
    if alerts_df.empty:
        st.info("No alerts found.")
    else:
        st.dataframe(alerts_df, use_container_width=True)

        # Optional: Show expanded alert details
        with st.expander("🔍 View Rule Details for First Alert"):
            first_alert = alerts_df.iloc[0]
            st.json({
                "rule_id": first_alert["rule_id"],
                "message": first_alert["message"],
                "log_ids": first_alert["related_log_ids"].split(",")
            })

# Optional: Logout
st.sidebar.markdown("---")
if st.sidebar.button("🔓 Logout"):
    st.session_state.clear()
    st.experimental_rerun()
