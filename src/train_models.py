
from pathlib import Path
import json
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "creditcard_cleaned.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "outputs" / "reports"
FIGURE_DIR = ROOT / "outputs" / "figures"

TARGET = "Class"
RANDOM_STATE = 42


def evaluate_model(model, X, y):
    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    return {
        "precision": float(
            precision_score(y, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y, predictions, zero_division=0)
        ),
        "f1_score": float(
            f1_score(y, predictions, zero_division=0)
        ),
        "average_precision": float(
            average_precision_score(y, probabilities)
        ),
        "roc_auc": float(roc_auc_score(y, probabilities)),
    }


def main():
    for folder in [MODEL_DIR, REPORT_DIR, FIGURE_DIR]:
        folder.mkdir(parents=True, exist_ok=True)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {DATA_PATH}. "
            "Run python src/clean_dataset.py first."
        )

    print("Loading cleaned dataset...")
    df = pd.read_csv(DATA_PATH)

    if TARGET not in df.columns:
        raise ValueError(f"Target column {TARGET!r} not found.")

    if df.isna().any().any():
        raise ValueError("Missing values found in cleaned data.")

    if df[TARGET].nunique() != 2:
        raise ValueError("Expected a binary classification target.")

    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)

    # Split 1: reserve 20% for the final test set.
    X_development, X_test, y_development, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            stratify=y,
            random_state=RANDOM_STATE,
        )
    )

    # Split 2: divide the remaining 80% into 60% train
    # and 20% validation of the full dataset.
    X_train, X_val, y_train, y_val = train_test_split(
        X_development,
        y_development,
        test_size=0.25,
        stratify=y_development,
        random_state=RANDOM_STATE,
    )

    print("\n========== DATA SPLITS ==========")
    print(f"Training rows:   {len(X_train):,}")
    print(f"Validation rows: {len(X_val):,}")
    print(f"Test rows:       {len(X_test):,}")
    print(f"Training fraud cases:   {int(y_train.sum())}")
    print(f"Validation fraud cases: {int(y_val.sum())}")
    print(f"Test fraud cases:       {int(y_test.sum())}")

    # Scale only Time and Amount. The scaler is fitted
    # inside each pipeline using training data only.
    preprocessor = ColumnTransformer(
        transformers=[
            ("scale_numeric", StandardScaler(), ["Time", "Amount"])
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )

    models = {
        "logistic_regression": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=200,
                        class_weight="balanced_subsample",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }

    validation_results = []

    for name, model in models.items():
        print(f"\nTraining {name}...")
        start = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - start

        metrics = evaluate_model(model, X_val, y_val)

        result = {
            "model": name,
            **metrics,
            "training_seconds": round(elapsed, 2),
        }
        validation_results.append(result)

        print(f"Validation results: {result}")

        joblib.dump(model, MODEL_DIR / f"{name}.joblib")

    # Select using validation average precision, not test results.
    validation_df = pd.DataFrame(validation_results)
    validation_df = validation_df.sort_values(
        "average_precision", ascending=False
    )

    validation_path = REPORT_DIR / "model_comparison_validation.csv"
    validation_df.to_csv(validation_path, index=False)

    best_name = str(validation_df.iloc[0]["model"])
    best_model = models[best_name]

    print("\nSelected model using validation average precision:")
    print(best_name)

    # Evaluate the selected model on the untouched test set.
    test_metrics = evaluate_model(best_model, X_test, y_test)
    test_probabilities = best_model.predict_proba(X_test)[:, 1]
    test_predictions = (test_probabilities >= 0.5).astype(int)

    matrix = confusion_matrix(
        y_test, test_predictions, labels=[0, 1]
    )

    report = {
        "selected_model": best_name,
        "selection_metric": "validation_average_precision",
        "decision_threshold": 0.5,
        "test_metrics": test_metrics,
        "test_confusion_matrix": {
            "labels": ["genuine_0", "fraud_1"],
            "matrix": matrix.tolist(),
        },
        "split_sizes": {
            "train": int(len(X_train)),
            "validation": int(len(X_val)),
            "test": int(len(X_test)),
        },
        "random_state": RANDOM_STATE,
    }

    report_path = REPORT_DIR / "test_evaluation.json"
    report_path.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8",
    )

    predictions_df = pd.DataFrame(
        {
            "actual_class": y_test.to_numpy(),
            "fraud_probability": test_probabilities,
            "predicted_class_at_0_5": test_predictions,
        }
    )
    predictions_df.to_csv(
        REPORT_DIR / "test_predictions.csv",
        index=False,
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["Genuine", "Fraud"],
    )
    display.plot(values_format="d")
    plt.title(f"Test Confusion Matrix: {best_name}")
    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR / "test_confusion_matrix.png",
        dpi=160,
    )
    plt.close()

    print("\n========== FINAL TEST RESULTS ==========")
    print(json.dumps(test_metrics, indent=4))
    print("\nConfusion matrix:")
    print(matrix)
    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            test_predictions,
            labels=[0, 1],
            target_names=["Genuine", "Fraud"],
            zero_division=0,
        )
    )

    print("\nSaved validation comparison:", validation_path)
    print("Saved test evaluation:", report_path)
    print("Saved test predictions:", REPORT_DIR / "test_predictions.csv")
    print("Saved confusion matrix:", FIGURE_DIR / "test_confusion_matrix.png")
    print("Saved models:", MODEL_DIR)
    print("\nModel training completed successfully.")


if __name__ == "__main__":
    main()
