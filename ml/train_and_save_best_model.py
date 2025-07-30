import sys
import os
import joblib
sys.path.append('../')

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import IsolationForest, RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from loglizer import dataloader, preprocessing

struct_log = './log_data/HDFS/HDFS_100k.log_structured.csv'
label_file = './log_data/HDFS/anomaly_label.csv'

if __name__ == '__main__':
    # Load and transform data
    (x_train, y_train), (x_test, y_test) = dataloader.load_HDFS(
        struct_log, label_file=label_file, window='session',
        train_ratio=0.5, split_type='uniform'
    )

    feature_extractor = preprocessing.FeatureExtractor()
    x_train = feature_extractor.fit_transform(x_train, term_weighting='tf-idf')
    x_test = feature_extractor.transform(x_test)

    models = [
        DecisionTreeClassifier(max_depth=10, class_weight='balanced', random_state=42),
        IsolationForest(contamination=0.1, random_state=42),
        LinearSVC(class_weight='balanced', random_state=42),
        LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
        RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
        GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    ]

    model_types = [
        "DecisionTree", "IsolationForest", "LinearSVC", "LogisticRegression", "RandomForest", "GradientBoosting"
    ]

    best_f1 = 0
    best_model = None
    best_model_name = None

    for model, name in zip(models, model_types):
        try:
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            report = classification_report(y_test, y_pred, digits=4, output_dict=True)
            f1 = report['1']['f1-score'] if '1' in report else 0.0
            print(f"F1-score ({name}): {f1:.4f}")
            print(classification_report(y_test, y_pred, digits=4))

            if f1 > best_f1:
                best_f1 = f1
                best_model = model
                best_model_name = name
        except Exception as e:
            print(f"Error with model {name}: {e}")

    # Save best model
    if best_model:
        os.makedirs("saved_models", exist_ok=True)
        joblib.dump(best_model, f"saved_models/{best_model_name}.pkl")
        print(f"Saved best model: {best_model_name} with F1 {best_f1:.4f}")
