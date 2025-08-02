# LogTrack — Log Anomaly and Alerting System

## Overview

LogTrack is a lightweight log anomaly detection and alerting tool designed to monitor service logs and detect suspicious patterns in real time.

## Features

- Ingest logs from files into SQLite DB via CLI  
- Define alerting rules (keyword, repeated user failures, rate spike)  
- Run detection rules on logs via CLI, write alerts to DB  
- Simple Streamlit web UI for viewing alerts and managing rules  
- **Z-score based anomaly detection** (statistical spike detection; optional, enabled via CLI flag)  
- **XGBoost ML anomaly detection** (ML based point anomaly detection; optional, enabled via CLI flag) 

## Z-score Anomaly Detection

Detects abnormal spikes in log volume by comparing the current time window’s log count against a baseline of previous windows using the statistical z-score. Alerts trigger when the z-score exceeds the configured threshold.

**Parameters:**

- `service`: the service to monitor  
- `window_minutes`: length of each time window  
- `baseline_windows`: number of historical windows used as baseline  
- `threshold`: z-score threshold for triggering alerts (default 3.0)  

## XGBOOST ML Anomaly Detection

Detects abnormal log entries based upon a pre-trained XGBoost model. Such a model is constructed using a dataset of HDFS (Hadoop Distributed File System) logs, as found in Loglizer [He et al, 2016].

### Training the XGBoost Model
To train and save the XGBoost model, there are two options.

The following command does not account for Block IDs, a feature within the HDFS logs of the dataset.
```
python3 ml/train_and_save_xgboost_model.py
```

To use Block IDs as a feature, use the command
```
python3 ml/train_and_save_xgboost_model.py --extract_block_id
```
The appropriate command to use depends upon the nature of the logs being ingested.
Both commands save the XGBoost model and log feature extractor as .pkl files to the directories saved_models/ and saved_feature_extractor/, respectively.

## Installation

```
git clone https://github.com/adityapatwardhan1/logtrack.git
cd logtrack
pip install -r requirements.txt
```

## Initialization
Consider the following as seeding scripts to be run once to initialize the system.

Initialize database (from root):
```
python3 db/init_db.py
```

Add rules:
```
python3 migrate_rules.py
```
This adds rules initialized from a starter JSON file.

Create users:
```
python3 -m cli.create_user name password
```

## Usage
Usage
1. Ingest Logs

Ingest logs from a log file into the SQLite database.

The file types supported include:

- Generic log files with a parseable username, timestamp, service, and message (see sample.log)
- HDFS files stored in CSV format 
- JSON files storing entries containing username, timestamp, service, and message 

The relevant commands are as follows:

```
python3 -m cli.ingest_logs path/to/logfile.log --db-path path/to/logtrack.db
```

```
python3 -m cli.ingest_logs path/to/hdfs_logfile.csv --db-path path/to/logtrack.db
```

```
python3 -m cli.ingest_logs path/to/json_logfile.json --db-path path/to/logtrack.db
```

--db-path is optional; defaults to logtrack.db.

2. Run Detection Rules

Run configured alerting rules against the logs and write alerts to DB.

```
python3 -m cli.run_detection --db-path path/to/logtrack.db [--zscore] [--ml]
```

Use --zscore to enable z-score anomaly detection rules; use --ml to enable XGBoost-based point anomaly detection.
(See the section on ML Anomaly Detection to train/create the model.)

3. Streamlit UI

Run the Streamlit dashboard UI:

```
streamlit run dashboard/app.py
```

Open your browser at the URL Streamlit provides (usually http://localhost:8501).

## References
Shilin He, Jieming Zhu, Pinjia He, Michael R. Lyu. Experience Report: System Log Analysis for Anomaly Detection, IEEE International Symposium on Software Reliability Engineering (ISSRE), 2016. [Bibtex][中文版本] (ISSRE Most Influential Paper)