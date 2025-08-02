import pandas as pd
import numpy as np
import os
import joblib
import argparse 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, FeatureHasher
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report
from scipy.sparse import hstack
from xgboost import XGBClassifier

# Parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Train XGBoost classifier on log data.")
    parser.add_argument(
        '--extract_block_id', action='store_true',
        help="Whether to extract BlockId and calculate block count for each BlockId."
    )
    return parser.parse_args()

# Load data
log_file = './log_data/HDFS/HDFS_100k.log_structured.csv'
label_file = './log_data/HDFS/anomaly_label.csv'
logs_df = pd.read_csv(log_file)
labels_df = pd.read_csv(label_file)

# Extract BlockId and merge
logs_df['BlockId'] = logs_df['Content'].str.extract(r'(blk_-?\d+)')
merged_df = pd.merge(logs_df, labels_df, on='BlockId', how='inner')

# Timestamp and IP extraction
merged_df['timestamp'] = pd.to_datetime(
    merged_df['Date'].astype(str) + merged_df['Time'].astype(str),
    format='%y%m%d%H%M%S', errors='coerce'
)
merged_df['user'] = merged_df['Content'].str.extract(r'src: /(\d+\.\d+\.\d+\.\d+)')

# Parse the arguments
args = parse_args()

# Add optional feature: event count per BlockId based on CLI argument
if args.extract_block_id:
    block_counts = merged_df['BlockId'].value_counts().to_dict()
    merged_df['block_count'] = merged_df['BlockId'].map(block_counts)
else:
    merged_df['block_count'] = 0  # Set all block counts to zero

# Final columns (no block_count)
data_df = merged_df[['timestamp', 'Component', 'Content', 'user', 'Label']].copy()
data_df.columns = ['timestamp', 'service', 'message', 'user', 'label']

# Extract BlockId from message if present (e.g., blk_-1234567)
data_df['BlockId'] = data_df['message'].str.extract(r'(blk_-?\d+)')

# Count frequency of each block
block_counts = data_df['BlockId'].value_counts().to_dict()

# Save block count mapping for consistency
os.makedirs("saved_feature_extractor", exist_ok=True)
joblib.dump(block_counts, "saved_feature_extractor/block_count_mapping.pkl")

# Apply count as a feature
data_df['block_count'] = data_df['BlockId'].map(block_counts).fillna(0).astype(int)

# Count frequency of each block (only where block exists)
block_counts = data_df['BlockId'].value_counts().to_dict()
print("block_counts head =", data_df['BlockId'].value_counts().head(10))
print("------------------------")
print("block_counts tail =", data_df['BlockId'].value_counts().tail(10))

# Save block counts mapping for reuse
os.makedirs("saved_feature_extractor", exist_ok=True)
joblib.dump(block_counts, "saved_feature_extractor/block_count_mapping.pkl")

# Map count to each row, fill missing with 0
data_df['block_count'] = data_df['BlockId'].map(block_counts).fillna(0).astype(int)

# Drop NA and remap labels
data_df.dropna(subset=['message', 'label'], inplace=True)
print(data_df)
data_df.loc[:, 'label'] = data_df['label'].map({'Normal': 0, 'Anomaly': 1})

# Timestamp features
data_df.loc[:, 'hour'] = data_df['timestamp'].dt.hour
data_df.loc[:, 'dayofweek'] = data_df['timestamp'].dt.dayofweek

# TF-IDF on message
tfidf = TfidfVectorizer(ngram_range=(1, 3), max_features=10000)
X_text = tfidf.fit_transform(data_df['message'])

# Scale numeric timestamp + block_count
X_numeric = data_df[['hour', 'dayofweek', 'block_count']].astype(np.float32).values
scaler = StandardScaler()
X_numeric_scaled = scaler.fit_transform(X_numeric)

# One-hot encode service
service_enc = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
X_service = service_enc.fit_transform(data_df[['service']])

# Hash user field (needs list of lists)
user_data = data_df['user'].fillna("unknown").astype(str).apply(lambda x: [x]).tolist()
hasher = FeatureHasher(n_features=16, input_type='string')
X_user = hasher.transform(user_data)

# Combine features
X_combined = hstack([X_text, X_numeric_scaled, X_service, X_user])
# y = data_df['label'].values
y = data_df['label'].fillna(-1).astype(int).values

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y, test_size=0.5, stratify=y, random_state=42
)

# Use class ratio instead of SMOTE
imbalance_ratio = sum(y_train == 0) / sum(y_train == 1)

model = XGBClassifier(
    scale_pos_weight=imbalance_ratio,
    use_label_encoder=False,
    eval_metric='logloss',
    n_jobs=-1
)
model.fit(X_train, y_train)

# Threshold sweep
best_f1, best_threshold, reports = 0, 0.5, {}
y_proba = model.predict_proba(X_test)

best = 0
for i in range(1, 20):
    threshold = i / 20
    y_pred = (y_proba[:, 1] > threshold).astype(int)
    report = classification_report(y_test, y_pred, digits=4, output_dict=True)
    f1 = report.get('1', {}).get('f1-score', 0.0)
    print(f"Threshold: {threshold:.2f} — F1: {f1:.4f}")
    reports[threshold] = report
    if f1 > best_f1:
        best = i
        best_f1, best_threshold = f1, threshold

# Save artifacts
os.makedirs("saved_models", exist_ok=True)
os.makedirs("saved_feature_extractor", exist_ok=True)

joblib.dump(model, "saved_models/XGBoostClassifier.pkl")
joblib.dump(tfidf, "saved_feature_extractor/feature_extractor.pkl")
joblib.dump(scaler, "saved_feature_extractor/scaler.pkl")
joblib.dump(service_enc, "saved_feature_extractor/service_encoder.pkl")
joblib.dump(hasher, "saved_feature_extractor/user_hasher.pkl")

with open("saved_models/threshold.txt", "w") as f:
    f.write(str(best_threshold))

# Print final results
print(f"\nBest Threshold: {best_threshold:.2f}")
print(f"Best F1: {best_f1:.4f}")
print("Best report:")
print(reports[best_threshold])
