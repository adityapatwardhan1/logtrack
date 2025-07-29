# Imports
from sklearn.ensemble import IsolationForest
import numpy as np
import joblib

def train_model(X: np.ndarray) -> IsolationForest:
    """
    Train an IsolationForest model on the given feature matrix.

    :param X: A 2D array of features.
    :type X: np.ndarray

    :returns: Trained IsolationForest model.
    :rtype: IsolationForest
    """
    clf = IsolationForest(random_state=42)
    clf.fit(X)
    return clf


def save_model(model: IsolationForest, path: str = "ml/isolation_forest.joblib") -> None:
    """Saves the trained model to disk."""
    joblib.dump(model, path)


def load_model(path: str = "ml/isolation_forest.joblib") -> IsolationForest:
    """Loads a trained model from disk."""
    try:
        return joblib.load(path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Failed to load file {path}")
    except:
        raise Exception(f"Failed to load model from {path}")
