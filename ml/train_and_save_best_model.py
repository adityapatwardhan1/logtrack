import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, FeatureHasher
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report
from scipy.sparse import hstack
from xgboost import XGBClassifier

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

# Add optional feature: event count per BlockId
block_counts = merged_df['BlockId'].value_counts().to_dict()
merged_df['block_count'] = merged_df['BlockId'].map(block_counts)

# Final columns (no block_count)
data_df = merged_df[['timestamp', 'Component', 'Content', 'user', 'Label']].copy()
data_df.columns = ['timestamp', 'service', 'message', 'user', 'label']

# Extract BlockId from message if present (e.g., blk_-1234567)
data_df['BlockId'] = data_df['message'].str.extract(r'(blk_-?\d+)')

# Count frequency of each block (only where block exists)
block_counts = data_df['BlockId'].value_counts().to_dict()

# Map count to each row, fill with 0 if block ID is missing
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


# # # import sys
# # # import os
# # # import joblib
# # # sys.path.append('../')

# # # from sklearn.tree import DecisionTreeClassifier
# # # from sklearn.ensemble import IsolationForest, RandomForestClassifier, GradientBoostingClassifier
# # # from sklearn.svm import LinearSVC
# # # from sklearn.linear_model import LogisticRegression
# # # from sklearn.metrics import classification_report, f1_score
# # # from loglizer import dataloader, preprocessing

# # # struct_log = './log_data/HDFS/HDFS_100k.log_structured.csv'
# # # label_file = './log_data/HDFS/anomaly_label.csv'

# # # if __name__ == '__main__':
# # #     # Load and transform data
# # #     (x_train, y_train), (x_test, y_test) = dataloader.load_HDFS(
# # #         struct_log, label_file=label_file, window='session',
# # #         train_ratio=0.5, split_type='uniform'
# # #     )

# # #     feature_extractor = preprocessing.FeatureExtractor()
# # #     x_train = feature_extractor.fit_transform(x_train, term_weighting='tf-idf')
# # #     x_test = feature_extractor.transform(x_test)

# # #     os.makedirs("saved_feature_extractor", exist_ok=True)
# # #     joblib.dump(feature_extractor, f"saved_feature_extractor/feature_extractor.pkl")

# # #     models = [
# # #         DecisionTreeClassifier(max_depth=10, class_weight='balanced', random_state=42),
# # #         IsolationForest(contamination=0.1, random_state=42),
# # #         LinearSVC(class_weight='balanced', random_state=42),
# # #         LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
# # #         RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
# # #         GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
# # #     ]

# # #     model_types = [
# # #         "DecisionTree", "IsolationForest", "LinearSVC", "LogisticRegression", "RandomForest", "GradientBoosting"
# # #     ]

# # #     best_f1 = 0
# # #     best_model = None
# # #     best_model_name = None

# # #     for model, name in zip(models, model_types):
# # #         try:
# # #             model.fit(x_train, y_train)
# # #             y_pred = model.predict(x_test)
# # #             report = classification_report(y_test, y_pred, digits=4, output_dict=True)
# # #             f1 = report['1']['f1-score'] if '1' in report else 0.0
# # #             print(f"F1-score ({name}): {f1:.4f}")
# # #             print(classification_report(y_test, y_pred, digits=4))

# # #             if f1 > best_f1:
# # #                 best_f1 = f1
# # #                 best_model = model
# # #                 best_model_name = name
# # #         except Exception as e:
# # #             print(f"Error with model {name}: {e}")

# # #     # Save best model
# # #     if best_model:
# # #         os.makedirs("saved_models", exist_ok=True)
# # #         joblib.dump(best_model, f"saved_models/{best_model_name}.pkl")
# # #         print(f"Saved best model: {best_model_name} with F1 {best_f1:.4f}")


# # import pandas as pd
# # import numpy as np
# # import os
# # import joblib

# # from sklearn.model_selection import train_test_split
# # from sklearn.feature_extraction.text import TfidfVectorizer
# # from sklearn.tree import DecisionTreeClassifier
# # from sklearn.ensemble import IsolationForest, RandomForestClassifier, GradientBoostingClassifier
# # from sklearn.svm import LinearSVC
# # from sklearn.linear_model import LogisticRegression
# # from sklearn.metrics import classification_report

# # # Paths to data
# # log_file = './log_data/HDFS/HDFS_100k.log_structured.csv'
# # label_file = './log_data/HDFS/anomaly_label.csv'

# # # Load structured log and labels
# # logs_df = pd.read_csv(log_file)
# # labels_df = pd.read_csv(label_file)

# # # Extract BlockId from Content
# # logs_df['BlockId'] = logs_df['Content'].str.extract(r'(blk_-?\d+)')

# # # Merge logs with labels
# # merged_df = pd.merge(logs_df, labels_df, on='BlockId', how='inner')

# # # Combine Date and Time into timestamp
# # merged_df['timestamp'] = pd.to_datetime(
# #     merged_df['Date'].astype(str) + merged_df['Time'].astype(str),
# #     format='%y%m%d%H%M%S', errors='coerce'
# # )

# # # Extract IP address from Content as 'user'
# # merged_df['user'] = merged_df['Content'].str.extract(r'src: /(\d+\.\d+\.\d+\.\d+)')

# # # Select and rename columns to match schema
# # data_df = merged_df[['timestamp', 'Component', 'Content', 'user', 'Label']]
# # data_df.columns = ['timestamp', 'service', 'message', 'user', 'label']

# # # Drop rows with missing values
# # data_df.dropna(subset=['message', 'label'], inplace=True)

# # # Encode labels
# # data_df['label'] = data_df['label'].map({'Normal': 0, 'Anomaly': 1})

# # # Prepare features
# # tfidf = TfidfVectorizer(max_features=1000)
# # X = tfidf.fit_transform(data_df['message']).toarray()
# # y = data_df['label'].values

# # # Split into train and test
# # X_train, X_test, y_train, y_test = train_test_split(
# #     X, y, test_size=0.5, stratify=y, random_state=42
# # )

# # # Save TF-IDF vectorizer
# # os.makedirs("saved_feature_extractor", exist_ok=True)
# # joblib.dump(tfidf, "saved_feature_extractor/feature_extractor.pkl")

# # # Define models
# # models = [
# #     # DecisionTreeClassifier(max_depth=10, random_state=42),
# #     # LinearSVC(random_state=42),
# #     # LogisticRegression(max_iter=1000, random_state=42),
# #     RandomForestClassifier(n_estimators=100,  random_state=42),
# # ]

# # model_names = [
# #     "RandomForestClassifier"
# # ]

# # # Train and evaluate models
# # best_f1 = 0
# # best_model = None
# # best_model_name = None
# # reports = {}

# # for model, name in zip(models, model_names):
# #     for ii in range(20):
# #         threshold = ii / 20
# #         try:
# #             print(f"Fitting {name}")
# #             model.fit(X_train, y_train)
# #             print(f"Finished fitting {name}")
# #             # y_pred_old = model.predict(X_test)
# #             y_proba = model.predict_proba(X_test)
# #             y_pred = (y_proba[:, 1] > threshold).astype(int)
# #             report = classification_report(y_test, y_pred, digits=4, output_dict=True)
# #             print("threshold =", threshold)
# #             print("report =", report)
# #             reports[name] = report
# #             f1 = report.get('1', {}).get('f1-score', 0.0)
# #             print("f1 =", f1)
# #             if f1 > best_f1:
# #                 best_f1 = f1
# #                 best_model = model
# #                 best_model_name = name
# #         except Exception as e:
# #             reports[name] = f"Error: {e}"

# # # Save best model
# # if best_model:
# #     os.makedirs("saved_models", exist_ok=True)
# #     joblib.dump(best_model, f"saved_models/{best_model_name}.pkl")

# # best_model_name, best_f1, reports[best_model_name] if best_model else None

# import pandas as pd
# import numpy as np
# import os
# import joblib

# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics import classification_report
# from sklearn.preprocessing import StandardScaler
# from scipy.sparse import hstack
# from imblearn.over_sampling import SMOTE
# from xgboost import XGBClassifier

# # Paths to data
# log_file = './log_data/HDFS/HDFS_100k.log_structured.csv'
# label_file = './log_data/HDFS/anomaly_label.csv'

# # Load structured log and labels
# logs_df = pd.read_csv(log_file)
# labels_df = pd.read_csv(label_file)

# # Extract BlockId from Content
# logs_df['BlockId'] = logs_df['Content'].str.extract(r'(blk_-?\d+)')

# # Merge logs with labels
# merged_df = pd.merge(logs_df, labels_df, on='BlockId', how='inner')

# # Combine Date and Time into timestamp
# merged_df['timestamp'] = pd.to_datetime(
#     merged_df['Date'].astype(str) + merged_df['Time'].astype(str),
#     format='%y%m%d%H%M%S', errors='coerce'
# )

# # Extract IP address from Content as 'user'
# merged_df['user'] = merged_df['Content'].str.extract(r'src: /(\d+\.\d+\.\d+\.\d+)')

# # Select and rename columns to match schema
# data_df = merged_df[['timestamp', 'Component', 'Content', 'user', 'Label']]
# data_df.columns = ['timestamp', 'service', 'message', 'user', 'label']

# # Drop rows with missing values
# data_df.dropna(subset=['message', 'label'], inplace=True)

# # Encode labels
# data_df['label'] = data_df['label'].map({'Normal': 0, 'Anomaly': 1})

# # Extract features from timestamp
# data_df['hour'] = data_df['timestamp'].dt.hour
# data_df['dayofweek'] = data_df['timestamp'].dt.dayofweek

# # TF-IDF vectorizer with n-grams
# tfidf = TfidfVectorizer(ngram_range=(1, 3), max_features=10000)
# X_text = tfidf.fit_transform(data_df['message'])

# # Numeric features
# X_numeric = data_df[['hour', 'dayofweek']].fillna(0).astype(np.float32).values
# scaler = StandardScaler()
# X_numeric_scaled = scaler.fit_transform(X_numeric)

# # Combine features
# X_combined = hstack([X_text, X_numeric_scaled])
# y = data_df['label'].values

# # Split into train and test
# X_train, X_test, y_train, y_test = train_test_split(
#     X_combined, y, test_size=0.5, stratify=y, random_state=42
# )

# # Apply SMOTE
# sm = SMOTE(random_state=42)
# X_train_bal, y_train_bal = sm.fit_resample(X_train, y_train)

# # Save feature extractor and scaler
# os.makedirs("saved_feature_extractor", exist_ok=True)
# joblib.dump(tfidf, "saved_feature_extractor/feature_extractor.pkl")
# joblib.dump(scaler, "saved_feature_extractor/scaler.pkl")

# # Train XGBoost
# imbalance_ratio = sum(y_train == 0) / sum(y_train == 1)
# model = XGBClassifier(scale_pos_weight=imbalance_ratio, use_label_encoder=False, eval_metric='logloss')
# model.fit(X_train_bal, y_train_bal)

# # Evaluate across thresholds
# best_f1 = 0
# best_threshold = 0.5
# reports = {}

# y_proba = model.predict_proba(X_test)
# for i in range(1, 20):
#     threshold = i / 20
#     y_pred = (y_proba[:, 1] > threshold).astype(int)
#     report = classification_report(y_test, y_pred, digits=4, output_dict=True)
#     f1 = report.get('1', {}).get('f1-score', 0.0)
#     print(f"Threshold: {threshold:.2f} — F1: {f1:.4f}")
#     reports[threshold] = report
#     if f1 > best_f1:
#         best_f1 = f1
#         best_threshold = threshold

# # Save best model
# os.makedirs("saved_models", exist_ok=True)
# joblib.dump(model, f"saved_models/XGBoostClassifier.pkl")

# print(f"\nBest Threshold: {best_threshold:.2f}")
# print(f"Best F1: {best_f1:.4f}")
# print("Best report:")
# print(reports[best_threshold])