from sklearn.ensemble import IsolationForest
from ml.model import load_model
from ml.features import extract_features
import json
import sys


def detect_anomalies(logs: list[dict], model: IsolationForest = None) -> list[dict]:
    """
    Run inference on incoming logs using the trained IsolationForest model.

    :param logs: List of log entries.
    :type logs: list[dict]

    :returns: List of logs annotated with an 'anomaly' boolean.
    :rtype: list[dict]
    """
    if not model:
        model = load_model()
    features = extract_features(logs)
    scores = model.decision_function(features)
    predictions = model.predict(features)

    # Annotate list of logs as normal/anomolous
    annotated = []
    for i in range(len(logs)):
        logs[i]["anomaly"] = (predictions[i] == -1)
        logs[i]["anomaly_score"] = float(scores[i])
        annotated.append(logs[i])
    return annotated

if __name__ == "__main__":
    with open(sys.argv[1], "r") as f:
        logs = [json.loads(line) for line in f]
    
    output = detect_anomalies(logs)
    for log in output:
        print(json.dumps(log))