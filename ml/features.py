# Imports
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import TruncatedSVD
from datetime import datetime

def extract_features(logs: list[dict]) -> np.ndarray:
    """
    Convert a list of log dictionaries into numerical feature vectors.

    :param logs: List of dictionaries, each containing keys like 'timestamp', 'service', 'message', etc.
    :type logs: list[dict]

    :returns: 2D NumPy array of shape (n_logs, n_features), suitable for model input.
    :rtype: np.ndarray
    """
    messages = [log.get("message", "") for log in logs]
    services = [log.get("service", "unknown") for log in logs]
    levels = [log.get("level", "INFO") for log in logs]
    hours = [datetime.fromisoformat(log["timestamp"]).hour if "timestamp" in log else 0 for log in logs]

    # Vectorize
    vectorizer = TfidfVectorizer(stop_words='english', max_features=200)
    messages_features = vectorizer.fit_transform(messages)
    truncated_svd = TruncatedSVD(n_components=20, random_state=42)
    messages_features = truncated_svd.fit_transform(messages_features)
    
    # One-hot encode services/levels  
    one_hot_encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
    one_hot_features = one_hot_encoder.fit_transform(np.array(list(zip(services, levels))))
    
    hours_features = np.array(hours).reshape(-1, 1)

    # Combine and return
    all_features = np.hstack([messages_features, one_hot_features, hours_features])
    return all_features

