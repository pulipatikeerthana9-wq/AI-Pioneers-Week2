"""Reproducible Week 2 supervised-learning model comparison.

Datasets are bundled with scikit-learn; all metrics are measured on held-out test data.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

SEED = 42
TEST_SIZE = 0.20
OUTPUT_DIR = Path(__file__).resolve().parent / "results"


def run_regression() -> pd.DataFrame:
    data = load_diabetes()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="disease_progression")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED
    )
    models = {
        "Linear Regression": LinearRegression(),
        "K-Nearest Neighbors": make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=5)),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=-1),
    }
    rows = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        rows.append({
            "model": name,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "MAE": mean_absolute_error(y_test, pred),
            "RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
            "R2": r2_score(y_test, pred),
        })
    result = pd.DataFrame(rows)
    assert result.shape[0] == 3 and np.isfinite(result.select_dtypes("number")).all().all()
    return result


def run_classification() -> pd.DataFrame:
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    # scikit-learn's original target: 0=malignant, 1=benign. Positive class is explicitly
    # set to malignant below by using malignant as label 1 for clinically intuitive recall.
    y = pd.Series((data.target == 0).astype(int), name="malignant")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )
    models = {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, random_state=SEED)),
        "Decision Tree": DecisionTreeClassifier(random_state=SEED),
        "K-Nearest Neighbors": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1),
    }
    rows = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        malignant_index = list(model.classes_).index(1)
        score = model.predict_proba(X_test)[:, malignant_index]
        rows.append({
            "model": name,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "accuracy": accuracy_score(y_test, pred),
            "precision_malignant": precision_score(y_test, pred, zero_division=0),
            "recall_malignant": recall_score(y_test, pred, zero_division=0),
            "f1_malignant": f1_score(y_test, pred, zero_division=0),
            "roc_auc_malignant": roc_auc_score(y_test, score),
        })
    result = pd.DataFrame(rows)
    assert result.shape[0] == 4 and np.isfinite(result.select_dtypes("number")).all().all()
    assert set(y.unique()) == {0, 1}
    return result


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    regression = run_regression()
    classification = run_classification()
    regression.to_csv(OUTPUT_DIR / "regression_metrics.csv", index=False, float_format="%.6f")
    classification.to_csv(OUTPUT_DIR / "classification_metrics.csv", index=False, float_format="%.6f")
    metadata = {
        "random_seed": SEED,
        "test_size": TEST_SIZE,
        "sklearn_version": sklearn.__version__,
        "regression_dataset": "scikit-learn Diabetes (442 rows, 10 features)",
        "classification_dataset": "scikit-learn Breast Cancer Wisconsin (569 rows, 30 features)",
        "classification_positive_class": "malignant (re-mapped to 1)",
        "note": "Holdout benchmark results; not clinical guidance or a guarantee of generalization.",
    }
    (OUTPUT_DIR / "experiment_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print("Regression metrics (lower MAE/RMSE and higher R2 are better):")
    print(regression.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nClassification metrics (malignant is positive class):")
    print(classification.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
