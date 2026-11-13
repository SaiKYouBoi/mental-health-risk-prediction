import json

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from .preprocessing import root, load_data, build_preprocessor


def train_models():
    X_train, X_test, y_train, y_test = load_data()
    preprocessor = build_preprocessor(X_train)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=1),
        "SVM": SVC(kernel="rbf")
    }

    pipelines = {}
    for name, model in models.items():
        pipelines[name] = Pipeline([
            ("preprocessor", clone(preprocessor)),
            ("model", model)
        ])

    pipelines["SVM"] = CalibratedClassifierCV(
        estimator=pipelines["SVM"], cv=5, ensemble=False
    )

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision_weighted",
        "recall": "recall_weighted",
        "f1": "f1_weighted",
        "roc_auc": "roc_auc_ovr_weighted"
    }

    results = []
    for name, pipeline in pipelines.items():
        print(f"Comparing {name}", flush=True)
        scores = cross_validate(
            pipeline, X_train, y_train, cv=cv, scoring=scoring,
            return_train_score=True, error_score="raise"
        )
        results.append({
            "Model": name,
            "Accuracy": scores["test_accuracy"].mean(),
            "Precision": scores["test_precision"].mean(),
            "Recall": scores["test_recall"].mean(),
            "F1": scores["test_f1"].mean(),
            "ROC_AUC": scores["test_roc_auc"].mean(),
            "Train F1": scores["train_f1"].mean()
        })

    comparison = pd.DataFrame(results).set_index("Model").sort_values("F1", ascending=False)
    print(comparison.round(3))

    param_grids = {
        "Logistic Regression": {"model__C": [0.01, 0.1, 1, 10]},
        "SVM": {
            "estimator__model__C": [0.1, 1, 10],
            "estimator__model__gamma": ["scale", 0.01, 0.1]
        }
    }

    searches = {}
    tuned_results = []
    for name, params in param_grids.items():
        print(f"Tuning {name}", flush=True)
        search = GridSearchCV(
            pipelines[name], params, scoring="f1_weighted", cv=cv,
            refit=True, n_jobs=1, error_score="raise"
        )
        search.fit(X_train, y_train)
        searches[name] = search
        tuned_results.append({
            "Model": name,
            "Baseline CV F1": comparison.loc[name, "F1"],
            "Tuned CV F1": search.best_score_,
            "Best parameters": search.best_params_
        })

    tuning_comparison = pd.DataFrame(tuned_results).set_index("Model")
    print(tuning_comparison)

    final_name = tuning_comparison["Tuned CV F1"].idxmax()
    final_search = searches[final_name]
    final_model = final_search.best_estimator_

    model_dir = root / "models"
    report_dir = root / "reports"
    model_dir.mkdir(exist_ok=True)
    report_dir.mkdir(exist_ok=True)

    joblib.dump(final_model, model_dir / "mental_health_pipeline.joblib")
    comparison.to_csv(report_dir / "model_comparison.csv")
    tuning_comparison.to_csv(report_dir / "tuning_comparison.csv")

    metadata = {
        "model": final_name,
        "best_parameters": final_search.best_params_,
        "cv_weighted_f1": float(final_search.best_score_),
        "feature_columns": X_train.columns.tolist(),
        "classes": final_model.classes_.tolist(),
        "sklearn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "joblib_version": joblib.__version__,
        "split_random_state": 42
    }
    (model_dir / "model_metadata.json").write_text(json.dumps(metadata, indent=2))
    print("Saved model:", final_name)
    return final_model


if __name__ == "__main__":
    train_models()
