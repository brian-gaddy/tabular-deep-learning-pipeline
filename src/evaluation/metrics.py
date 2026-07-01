import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def evaluate_model(model, X_test, y_test, threshold=0.5):
    y_prob = model.predict(X_test).ravel()
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "roc_auc": roc_auc_score(y_test, y_prob),
        "pr_auc": average_precision_score(y_test, y_prob),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
    }

    print("\n=== Evaluation Metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    return metrics
