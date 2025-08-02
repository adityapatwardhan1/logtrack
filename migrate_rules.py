import json
import sqlite3

def migrate_rules(json_path="data/rules_config.json", db_path="logtrack.db"):
    with open(json_path, "r") as f:
        rules = json.load(f)

    con = sqlite3.connect(db_path)
    cur = con.cursor()
    for rule in rules:
        cur.execute("""
            INSERT OR REPLACE INTO rules (
                id, rule_type, service, keyword, message,
                threshold, window_minutes, window_seconds, max_idle_minutes,
                user_field, description, created_by
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            rule.get("id"),
            rule.get("rule_type"),
            rule.get("service"),
            rule.get("keyword"),
            rule.get("message"),
            rule.get("threshold"),
            rule.get("window_minutes"),
            rule.get("window_seconds"), 
            rule.get("max_idle_minutes"),
            rule.get("user_field"),
            rule.get("description", ""),
            None
        ))

    con.commit()
    con.close()
    print("Migration complete.")

if __name__ == "__main__":
    migrate_rules()
