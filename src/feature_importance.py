
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "random_forest.joblib"
DATA_PATH = ROOT / "data" / "processed" / "creditcard_cleaned.csv"
REPORT_DIR = ROOT / "outputs" / "reports"
FIGURE_DIR = ROOT / "outputs" / "figures"


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["Class"])

    model = joblib.load(MODEL_PATH)

    # Access the trained Random Forest inside the pipeline.
    classifier = model.named_steps["classifier"]
    preprocessor = model.named_steps["preprocessor"]

    feature_names = preprocessor.get_feature_names_out()
    importances = classifier.feature_importances_

    if len(feature_names) != len(importances):
        raise ValueError(
            "Feature names and importance values do not match."
        )

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    ).sort_values("importance", ascending=False)

    output_csv = REPORT_DIR / "feature_importance.csv"
    importance_df.to_csv(output_csv, index=False)

    top_features = importance_df.head(15).sort_values(
        "importance", ascending=True
    )

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(
        top_features["feature"],
        top_features["importance"],
    )
    ax.set_title("Random Forest Feature Importance")
    ax.set_xlabel("Relative importance")
    ax.set_ylabel("Feature")
    fig.tight_layout()

    output_png = FIGURE_DIR / "feature_importance.png"
    fig.savefig(output_png, dpi=160)
    plt.close(fig)

    print("\n========== FEATURE IMPORTANCE ==========")
    print(importance_df.head(15).to_string(index=False))
    print("\nSaved importance table:", output_csv)
    print("Saved importance chart:", output_png)
    print(
        "\nInterpretation note: These are global model importance "
        "scores, not proof of causation or an explanation of "
        "a specific transaction."
    )


if __name__ == "__main__":
    main()
