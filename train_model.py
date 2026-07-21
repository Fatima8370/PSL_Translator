"""
Trains a RandomForestClassifier on the combined landmark dataset and
saves it to disk for use by the live predictor later.
"""

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from load_dataset import load_full_dataset

MODEL_OUTPUT_PATH = "psl_classifier.joblib"


def train():
    X, y = load_full_dataset()

    unique_labels = set(y)
    if len(unique_labels) < 2:
        raise ValueError(
            f"Only {len(unique_labels)} label(s) found ({unique_labels}). "
            f"Need at least 2 distinct labels to train a classifier. "
            f"Collect more signs with data_collector.py first."
        )
    else:
        pass

    # Split so we can measure performance on data the model never trained on.
    # stratify=y ensures each label is proportionally represented in both
    # the train and test splits — important since some labels may have
    # more collected samples than others.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,   # number of trees; 200 is a solid starting point
        max_depth=None,      # let trees grow fully; we have low-dimensional,
                              # clean features so overfitting risk is manageable
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"Test accuracy: {accuracy:.4f}")
    print("\nPer-class performance:")
    print(classification_report(y_test, predictions))

    joblib.dump(model, MODEL_OUTPUT_PATH)
    print(f"\n[INFO] Model saved to {MODEL_OUTPUT_PATH}")


if __name__ == "__main__":
    train()