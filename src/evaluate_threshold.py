
from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "creditcard_cleaned.csv"
MODEL = ROOT / "models" / "random_forest.joblib"
THRESHOLDS = ROOT / "outputs" / "reports" / "threshold_analysis.csv"
REPORTS = ROOT / "outputs" / "reports"
FIGURES = ROOT / "outputs" / "figures"


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA)
    X = df.drop(columns=["Class"])
    y = df["Class"].astype(int)

    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y,
        test_size=0.20,
        stratify=y,
        random_state=42,
    )

    # Reproduce the same validation split used in training.
    _, _, _, _ = train_test_split(
        X_dev, y_dev,
        test_size=0.25,
        stratify=y_dev,
        random_state=42,
    )

    # Threshold selection uses validation results only.
    candidates = pd.read_csv(THRESHOLDS)
    eligible = candidates[candidates["recall"] >= 0.90]

    if eligible.empty:
        raise ValueError(
            "No threshold met the validation recall target."
        )

    selected = eligible.sort_values(
        ["precision", "f1_score", "threshold"],
        ascending=[False, False, False],
    ).iloc[0]

    threshold = float(selected["threshold"])

    model = joblib.load(MODEL)
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    ).ravel()

    metrics = {
        "model": "random_forest",
        "threshold_selected_on_validation": threshold,
        "test_precision": float(
            precision_score(y_test, predictions, zero_division=0)
        ),
        "test_recall": float(
            recall_score(y_test, predictions, zero_division=0)
        ),
        "test_f1_score": float(
            f1_score(y_test, predictions, zero_division=0)
        ),
        "test_average_precision": float(
            average_precision_score(y_test, probabilities)
        ),
        "test_roc_auc": float(
            roc_auc_score(y_test, probabilities)
        ),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "total_alerts": int(tp + fp),
        "test_transactions": int(len(y_test)),
    }

    output = REPORTS / "threshold_test_evaluation.json"
    output.write_text(json.dumps(metrics, indent=4), encoding="utf-8")

    print("\n========== HELD-OUT TEST RESULTS ==========")
    print(json.dumps(metrics, indent=4))
    print("\nSaved report:", output)
    print(
        "\nThe threshold was selected on validation data. "
        "Do not tune it using these test results."
    )


if __name__ == "__main__":
    main()
