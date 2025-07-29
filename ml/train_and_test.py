import os
import sys
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from loglizer import dataloader, preprocessing
import joblib

struct_log = './log_data/HDFS/HDFS_100k.log_structured.csv'
label_file = './log_data/HDFS/anomaly_label.csv'
model_dir = './ml/models'
os.makedirs(model_dir, exist_ok=True)

def evaluate(y_true, y_pred):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return precision, recall, f1

def find_best_threshold(scores, y_true):
    """
    Find threshold on decision_function scores that maximizes F1.
    Scores: higher means more normal, lower means more anomalous.
    """
    best_f1 = 0
    best_threshold = None
    for thresh in np.linspace(scores.min(), scores.max(), 100):
        preds = (scores < thresh).astype(int)
        f1 = f1_score(y_true, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
    return best_threshold

def main():
    # 1. Load data
    print("Loading structured logs and labels...")
    (x_train, y_train), (x_test, y_test) = dataloader.load_HDFS(
        struct_log,
        label_file=label_file,
        window='session',
        train_ratio=0.5,
        split_type='uniform',
    )

    # 2. Feature extraction
    feature_extractor = preprocessing.FeatureExtractor()
    x_train = feature_extractor.fit_transform(x_train)
    x_test = feature_extractor.transform(x_test)

    # 3. Train IsolationForest only on normal data
    x_train_normal = x_train[y_train == 0]
    print(f"Training IsolationForest on {x_train_normal.shape[0]} normal samples...")

    clf = IsolationForest(n_estimators=100, contamination='auto', random_state=42)
    clf.fit(x_train_normal)

    # 4. Get train decision scores & find best threshold
    train_scores = clf.decision_function(x_train)
    best_threshold = find_best_threshold(train_scores, y_train)
    print(f"Best threshold on train data: {best_threshold:.4f}")

    # 5. Predict using threshold on train and test
    train_preds = (train_scores < best_threshold).astype(int)
    test_scores = clf.decision_function(x_test)
    test_preds = (test_scores < best_threshold).astype(int)

    # 6. Evaluate and print results
    print("\nTrain classification report:")
    print(classification_report(y_train, train_preds, digits=4))

    print("Test classification report:")
    print(classification_report(y_test, test_preds, digits=4))

    # 7. Save model and feature extractor
    joblib.dump(clf, os.path.join(model_dir, "isolation_forest.pkl"))
    joblib.dump(feature_extractor, os.path.join(model_dir, "feature_extractor.pkl"))
    print(f"\n✅ Model and feature extractor saved to '{model_dir}'")

if __name__ == "__main__":
    main()
