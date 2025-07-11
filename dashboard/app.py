# dashboard/app.py
import streamlit as st
import sqlite3
import pandas as pd
from pathlib import Path

DB_PATH = Path("logtrack.db")

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