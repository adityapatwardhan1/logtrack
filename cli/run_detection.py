import sys
import traceback
import argparse
import json
from sqlite3 import OperationalError
from core.rules_engine import evaluate_rules
from core.alert_manager import record_alert

def main():
    """Main script for running rules-based detection"""
    # Parse CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--db-path', default='logtrack.db',
                        help='Path to SQLite database file (default: logtrack.db)')
    parser.add_argument("--zscore", action="store_true", help="Enable z-score based anomaly detection")
    args = parser.parse_args()

    # Call evaluate_rules
    try:
        triggered_alerts = evaluate_rules(args.db_path)
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

    # Loop alerts -> record_alert
    for alert in triggered_alerts:
        recorded_alerts.append(alert)
        record_alert(alert, args.db_path)

    num_alerts_triggered = len(triggered_alerts)
    rules_triggered = set(alert["rule_id"] for alert in triggered_alerts)

    # Print summary
    print('Number of alerts triggered:', num_alerts_triggered)
    print('Rules triggered:', ', '.join(sorted(rules_triggered)))
    print('Alerts written to DB:')
    for alert in recorded_alerts:
        print(f"  - [{alert['rule_id']}] {alert['triggered_at']}: {alert['message']}")


if __name__ == '__main__':
    main()
