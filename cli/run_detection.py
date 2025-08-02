import sys
import traceback
import argparse
import joblib
import os
import pandas as pd
import sqlite3
from sqlite3 import OperationalError
from core.rules_engine import evaluate_rules
from core.alert_manager import record_alert
from loglizer import preprocessing
from loglizer.dataloader import load_HDFS


def main():
    """Main script for running detection"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--db-path', default='logtrack.db',
                        help='Path to SQLite database file (default: logtrack.db)')
    parser.add_argument("--zscore", action="store_true", help="Enable z-score based anomaly detection")
    parser.add_argument("--ml", action="store_true")
    args = parser.parse_args()

    # Rules based detection
    try:
        ml_enabled = bool(args.ml and os.path.exists(args.ml))
        triggered_alerts = evaluate_rules(args.db_path, zscore_enabled=args.zscore, ml_enabled=ml_enabled)
    except OperationalError as e:
        print('An error occurred when connecting to the database file:')
        print(e)
        sys.exit(1)
    except Exception as e:
        print('An exception occurred while evaluating the provided rules:')
        print(e)
        traceback.print_exc()
        sys.exit(1)

    recorded_alerts = []
    for alert in triggered_alerts:
        recorded_alerts.append(alert)
        record_alert(alert, args.db_path)

    num_alerts_triggered = len(triggered_alerts)
    rules_triggered = set(alert["rule_id"] for alert in triggered_alerts)

    print('Number of alerts triggered:', num_alerts_triggered)
    print('Rules triggered:', ', '.join(sorted(rules_triggered)))
    print('Alerts written to DB:')
    for alert in recorded_alerts:
        print(f"  - [{alert['rule_id']}] {alert['triggered_at']}: {alert['message']}")

    


if __name__ == '__main__':
    main()
