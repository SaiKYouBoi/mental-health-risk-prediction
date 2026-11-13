"""Evaluate the saved model using notebook 4's test metrics and error analysis."""

import json

import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, ConfusionMatrixDisplay
)

from .preprocessing import root, load_data


def evaluate_model():
    X_train, X_test, y_train, y_test = load_data()
    model_dir = root / "models"
    report_dir = root / "reports"
    report_dir.mkdir(exist_ok=True)

    final_model = joblib.load(model_dir / "mental_health_pipeline.joblib")
    metadata = json.loads((model_dir / "model_metadata.json").read_text())

    y_pred = final_model.predict(X_test)
    y_proba = final_model.predict_proba(X_test)

    test_metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Weighted precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "Weighted recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "Weighted F1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "Weighted ROC-AUC": roc_auc_score(
            y_test, y_proba, labels=final_model.classes_,
            multi_class="ovr", average="weighted"
        )
    }
    print(pd.Series(test_metrics).round(3))

    risk_order = ["Low", "Medium", "High"]
    print(classification_report(y_test, y_pred, labels=risk_order, zero_division=0))
    report = classification_report(
        y_test, y_pred, labels=risk_order, output_dict=True, zero_division=0
    )

    plot = ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, labels=risk_order, cmap="Blues", colorbar=False
    )
    plt.title(f"{metadata['model']}: final test results")
    plt.tight_layout()
    plt.savefig(report_dir / "confusion_matrix.png")
    plt.close()
    pd.DataFrame(plot.confusion_matrix, index=risk_order, columns=risk_order).to_csv(
        report_dir / "confusion_matrix.csv", index_label="Actual"
    )

    test_review = X_test.copy()
    test_review["Actual"] = y_test
    test_review["Predicted"] = y_pred
    test_review["Correct"] = test_review["Actual"] == test_review["Predicted"]
    errors = test_review.loc[~test_review["Correct"]]
    print(errors.head(10))

    group_results = test_review.groupby("Gender")["Correct"].agg(["count", "mean"])
    group_results = group_results.rename(columns={"count": "Students", "mean": "Accuracy"})
    print(group_results.round(3))

    error_counts = {
        "High_false_negatives": int(((y_test == "High") & (y_pred != "High")).sum()),
        "High_false_positives": int(((y_test != "High") & (y_pred == "High")).sum())
    }
    print(error_counts)

    # Save results for the future Streamlit dashboard.
    test_review.to_csv(report_dir / "test_predictions.csv", index=False)
    errors.to_csv(report_dir / "prediction_errors.csv", index=False)
    group_results.to_csv(report_dir / "gender_performance.csv")
    (report_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2))
    (report_dir / "classification_report.json").write_text(json.dumps(report, indent=2))
    (report_dir / "error_counts.json").write_text(json.dumps(error_counts, indent=2))

    metadata["test_metrics"] = test_metrics
    (model_dir / "model_metadata.json").write_text(json.dumps(metadata, indent=2))
    return test_metrics


if __name__ == "__main__":
    evaluate_model()
