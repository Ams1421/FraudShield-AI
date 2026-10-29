
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"


def main():
    if not DATA_PATH.exists():
        print(f"\nDataset not found: {DATA_PATH}")
        print("Download creditcard.csv and place it in data/raw/.")
        return

    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    print("\n========== FRAUDSHIELD AI ==========")
    print("Dataset shape:", df.shape)

    print("\nFirst five rows:")
    print(df.head())

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDuplicate rows:", df.duplicated().sum())

    if "Class" in df.columns:
        print("\nClass distribution:")
        print(df["Class"].value_counts().sort_index())

        print("\nClass percentages:")
        print(
            (df["Class"].value_counts(normalize=True) * 100).round(4)
        )

    if "Amount" in df.columns:
        print("\nTransaction amount summary:")
        print(df["Amount"].describe())

    print("\nDataset inspection completed.")


if __name__ == "__main__":
    main()
