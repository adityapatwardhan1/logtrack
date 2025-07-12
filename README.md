# LogTrack — Log Anomaly and Alerting System

## Overview

LogTrack is a lightweight log anomaly detection and alerting tool designed to monitor service logs and detect suspicious patterns in real time.

## Features

- Ingest logs from files into SQLite DB via CLI  
- Define alerting rules (keyword, repeated user failures, rate spike)  
- Run detection rules on logs via CLI, write alerts to DB  
- Simple Streamlit web UI for viewing alerts and managing rules  
- **Z-score based anomaly detection** (statistical spike detection; optional, enabled via CLI flag)  

## Z-score Anomaly Detection

Detects abnormal spikes in log volume by comparing the current time window’s log count against a baseline of previous windows using the statistical z-score. Alerts trigger when the z-score exceeds the configured threshold.

**Parameters:**

- `service`: the service to monitor  
- `window_minutes`: length of each time window  
- `baseline_windows`: number of historical windows used as baseline  
- `threshold`: z-score threshold for triggering alerts (default 3.0)  

## Installation

```
git clone https://github.com/adityapatwardhan1/logtrack.git
cd logtrack
pip install -r requirements.txt
```

## Usage
Usage
1. Ingest Logs

Ingest logs from a log file into the SQLite database.

```
python3 -m cli.ingest_logs path/to/logfile.log --db-path path/to/logtrack.db
```

--db-path is optional; defaults to logtrack.db.

2. Run Detection Rules

Run configured alerting rules against the logs and write alerts to DB.

```
python3 -m cli.run_detection --db-path path/to/logtrack.db [--zscore]
```

Use --zscore to enable z-score anomaly detection rules.

3. Streamlit UI

Run the Streamlit dashboard UI:

```
streamlit run dashboard/app.py
```

Open your browser at the URL Streamlit provides (usually http://localhost:8501).

The UI allows non-admin users to view alerts and rules, while CLI is used for admin-level tasks like ingestion and detection.

## Other CLI commands

- Add users (admin only)

- Add rules (admin only)

- Ingest logs (admin only)
