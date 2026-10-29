
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "creditcard_cleaned.csv"
MODEL_PATH = ROOT / "models" / "random_forest.joblib"
REPORT_DIR = ROOT / "outputs" / "reports"
FIGURE_DIR = ROOT / "outputs" / "figures"

RANDOM_STATE = 42
TARGET_RECALL = 0.90


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["Class"])
    y = df["Class"].astype(int)

    # Recreate exactly the same stratified train/validation/test split.
    X_development, X_test, y_development, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_development,
        y_development,
        test_size=0.25,
        stratify=y_development,
        random_state=RANDOM_STATE,
    )

    model = joblib.load(MODEL_PATH)
    probabilities = model.predict_proba(X_val)[:, 1]

    results = []

    # Evaluate a range of candidate thresholds on validation data.
    thresholds = np.unique(
        np.append(np.arange(0.01, 1.00, 0.01), 0.50)
    )

    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_val, predictions, labels=[0, 1]
        ).ravel()

        results.append(
            {
                "threshold": float(threshold),
                "precision": precision_score(
                    y_val, predictions, zero_division=0
                ),
                "recall": recall_score(
                    y_val, predictions, zero_division=0
                ),
                "f1_score": f1_score(
                    y_val, predictions, zero_division=0
                ),
                "true_positives": int(tp),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_negatives": int(tn),
                "alerts": int(tp + fp),
            }
        )

    results_df = pd.DataFrame(results)
    results_path = REPORT_DIR / "threshold_analysis.csv"
    results_df.to_csv(results_path, index=False)

    # Choose the highest-precision candidate that meets the recall goal.
    eligible = results_df[
        results_df["recall"] >= TARGET_RECALL
    ]

    if eligible.empty:
        recommended = results_df.loc[
            results_df["recall"].idxmax()
        ]
        selection_note = (
            "No tested threshold met the target recall; "
            "selected a threshold with maximum validation recall."
        )
    else:
        recommended = eligible.sort_values(
            ["precision", "f1_score", "threshold"],
            ascending=[False, False, False],
        ).iloc[0]
        selection_note = (
            "Selected the highest-precision tested threshold "
            "meeting the validation recall target."
        )

    # Plot the validation precision-recall trade-off.
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(
        results_df["threshold"],
        results_df["precision"],
        label="Precision",
    )
    ax.plot(
        results_df["threshold"],
        results_df["recall"],
        label="Recall",
    )
    ax.plot(
        results_df["threshold"],
        results_df["f1_score"],
        label="F1-score",
    )
    ax.axhline(
        TARGET_RECALL,
        linestyle="--",
        label=f"Recall target ({TARGET_RECALL:.0%})",
    )
    ax.axvline(
        recommended["threshold"],
        linestyle=":",
        label=f"Candidate threshold ({recommended['threshold']:.2f})",
    )
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Metric")
    ax.set_ylim(0, 1.02)
    ax.set_title("Fraud Alert Threshold Analysis (Validation Set)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        FIGURE_DIR / "threshold_analysis.png",
        dpi=160,
    )
    plt.close(fig)

    report = [
        "FRAUDSHIELD AI - THRESHOLD ANALYSIS",
        "=" * 42,
        f"Validation rows: {len(X_val):,}",
        f"Validation fraud cases: {int(y_val.sum()):,}",
        f"Target recall: {TARGET_RECALL:.0%}",
        f"Candidate threshold: {recommended['threshold']:.2f}",
        f"Validation precision: {recommended['precision']:.4f}",
        f"Validation recall: {recommended['recall']:.4f}",
        f"Validation F1-score: {recommended['f1_score']:.4f}",
        f"Validation alerts: {int(recommended['alerts']):,}",
        f"Validation false positives: "
        f"{int(recommended['false_positives']):,}",
        f"Validation missed fraud: "
        f"{int(recommended['false_negatives']):,}",
        "",
        selection_note,
        "This is a candidate threshold, not a final production setting.",
        "The test set was not used to choose this threshold.",
    ]

    report_path = REPORT_DIR / "threshold_recommendation.txt"
    report_path.write_text("\n".join(report), encoding="utf-8")

    print("\n".join(report))
    print("\nSaved threshold table:", results_path)
    print("Saved threshold plot:", FIGURE_DIR / "threshold_analysis.png")
    print("Saved recommendation:", report_path)


if __name__ == "__main__":
    main()
