import json
import sys
from ml.features import extract_features
from ml.model import train_model, save_model

def load_logs(path: str) -> list[dict]:
    with open(path, "r") as f:
        return [json.loads(line) for line in f]

def train_and_score(train_path: str, test_path: str):
    train_logs = load_logs(train_path)
    test_logs = load_logs(test_path)

    X_train = extract_features(train_logs)
    X_test = extract_features(test_logs)

    model = train_model(X_train)
    save_model(model)

    preds = model.predict(X_test)
    outlier_fraction = (preds == -1).mean()
    print(f"Outlier rate on test set: {outlier_fraction:.3f}")

if __name__ == "__main__":
    train_and_score(sys.argv[1], sys.argv[2])
