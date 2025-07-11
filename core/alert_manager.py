from db.init_db import get_db_connection 

def record_alert(alert: dict, db_path: str) -> None:
    """
    Insert an alert into the alerts table
    :param alert: dict containing relevant info including rule_id, triggered_at, message and related_log_ids
    :type alert: dict 
    :param db_path: Path of database to write to
    :type db_path: str
    """
    con = get_db_connection(db_path=db_path)
    cur = con.cursor()
    cur.execute("INSERT INTO alerts (rule_id, triggered_at, message, related_log_ids)" \
    "VALUES (?,?,?,?)", (alert['rule_id'], alert['triggered_at'], alert['message'], ','.join(map(str, alert['related_log_ids']))))
    con.commit()
    con.close()    
