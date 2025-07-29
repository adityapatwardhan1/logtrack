#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_recall_fscore_support
from pathlib import Path

def load_windows(pkl_path):
    with open(pkl_path, "rb") as f:
        windows = pickle.load(f)
    return windows

def flatten_templates(windows, template_map):
    x = []
    y = []
    for window in windows:
        templates = [template_map.get(e["Content"], "") for e in window]
        labels = [e["Label"] for e in window]
        x.append(" ".join(templates))  # treat window as document
        y.append(1 if any(labels) else 0)
    return x, y

def train_iforest(train_x, test_x):
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(train_x)
    X_test = vectorizer.transform(test_x)

    model = IsolationForest(n_estimators=100, contamination='auto')
    model.fit(X_train)

    train_scores = model.decision_function(X_train)
    test_scores = model.decision_function(X_test)

    return train_scores, test_scores, model, vectorizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--structured_csv", type=str, required=True,
                        help="Path to structured log CSV with Content and EventTemplate columns.")
    parser.add_argument("--train_pkl", type=str, required=True,
                        help="Pickle file of training log windows (list of list of dicts).")
    parser.add_argument("--test_pkl", type=str, required=True,
                        help="Pickle file of testing log windows.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save trained model and vectorizer.")
    args = parser.parse_args()

    df = pd.read_csv(args.structured_csv)
    template_map = dict(zip(df["Content"], df["EventTemplate"]))

    train_windows = load_windows(args.train_pkl)
    test_windows = load_windows(args.test_pkl)

    train_x, train_y = flatten_templates(train_windows, template_map)
    test_x, test_y = flatten_templates(test_windows, template_map)

    train_scores, test_scores, model, vectorizer = train_iforest(train_x, test_x)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{args.output_dir}/iforest_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(f"{args.output_dir}/vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    # Optional: print eval if labels are available
    test_preds = (test_scores < 0).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(test_y, test_preds, average='binary')
    print(f"Test PRF: precision={precision:.3f}, recall={recall:.3f}, f1={f1:.3f}")

if __name__ == "__main__":
    main()
