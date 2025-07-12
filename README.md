# LogTrack — Log Anomaly and Alerting System

## Overview

LogTrack is a lightweight log anomaly detection and alerting tool designed for monitoring service logs and detecting suspicious patterns in real time.

## Features

- Keyword matching alerts  
- Rate spike detection  
- User-based failure thresholding  
- Z-score based anomaly detection (statistical spike detection) — **optional, enable via CLI flag**

## Z-score Anomaly Detection

This feature detects abnormal spikes in log volume by comparing the current window’s log count against a baseline of previous windows using statistical z-score. When the z-score exceeds the configured threshold, an alert is triggered.

### Parameters

- `service`: the service to monitor  
- `window_minutes`: size of each time window  
- `baseline_windows`: number of historical windows to compute baseline  
- `threshold`: z-score threshold for triggering alert (default 3.0)

## Installation

```bash
git clone https://github.com/adityapatwardhan1/logtrack.git
cd logtrack
pip install -r requirements.txt
```
