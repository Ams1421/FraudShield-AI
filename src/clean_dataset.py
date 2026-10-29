
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "creditcard.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "outputs" / "reports"

TARGET = "Class"


def main():
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Raw dataset not found: {RAW_PATH}"
        )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading original dataset...")
    df = pd.read_csv(RAW_PATH)

    # Validate the expected dataset structure.
    required_columns = {
        "Time", "Amount", TARGET,
        *[f"V{i}" for i in range(1, 29)],
    }

    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Required columns missing: {sorted(missing_columns)}"
        )

    if df.empty:
        raise ValueError("Dataset is empty.")

    if df.isna().any().any():
        raise ValueError(
            "Missing values found. Review them before cleaning."
        )

    if not set(df[TARGET].unique()).issubset({0, 1}):
        raise ValueError("Unexpected target labels found.")

    original_rows = len(df)

    # Investigate exact duplicate rows.
    duplicate_mask = df.duplicated(keep="first")
    duplicate_rows = df.loc[duplicate_mask].copy()

    duplicate_count = int(duplicate_mask.sum())
    duplicate_fraud_count = int(
        duplicate_rows[TARGET].sum()
    )
    duplicate_genuine_count = (
        duplicate_count - duplicate_fraud_count
    )

    print("\n========== DUPLICATE INVESTIGATION ==========")
    print("Original rows:", f"{original_rows:,}")
    print("Exact duplicate rows beyond first occurrence:",
          f"{duplicate_count:,}")
    print("Duplicate rows labelled genuine:",
          f"{duplicate_genuine_count:,}")
    print("Duplicate rows labelled fraud:",
          f"{duplicate_fraud_count:,}")

    # Document the cleaning policy.
    # Raw data remains unchanged. Exact duplicate records are
    # excluded from this separate modeling dataset.
    cleaned = df.drop_duplicates(keep="first").copy()

    removed_rows = original_rows - len(cleaned)

    # Validate the result.
    if cleaned.duplicated().any():
        raise RuntimeError("Exact duplicates remain.")

    if len(cleaned) + removed_rows != original_rows:
        raise RuntimeError("Row-count reconciliation failed.")

    if not set(cleaned[TARGET].unique()).issubset({0, 1}):
        raise RuntimeError("Invalid target labels after cleaning.")

    output_path = (
        PROCESSED_DIR / "creditcard_cleaned.csv"
    )
    cleaned.to_csv(output_path, index=False)

    # Save a class-level cleaning audit.
    audit = pd.DataFrame(
        {
            "metric": [
                "original_rows",
                "duplicate_rows_removed",
                "cleaned_rows",
                "original_fraud_rows",
                "cleaned_fraud_rows",
                "original_genuine_rows",
                "cleaned_genuine_rows",
                "missing_values_after_cleaning",
            ],
            "value": [
                original_rows,
                removed_rows,
                len(cleaned),
                int(df[TARGET].sum()),
                int(cleaned[TARGET].sum()),
                int((df[TARGET] == 0).sum()),
                int((cleaned[TARGET] == 0).sum()),
                int(cleaned.isna().sum().sum()),
            ],
        }
    )

    audit_path = REPORTS_DIR / "cleaning_audit.csv"
    audit.to_csv(audit_path, index=False)

    print("\n========== CLEANING RESULTS ==========")
    print("Rows removed:", f"{removed_rows:,}")
    print("Cleaned rows:", f"{len(cleaned):,}")
    print("Fraud rows remaining:", f"{int(cleaned[TARGET].sum()):,}")
    print("Genuine rows remaining:",
          f"{int((cleaned[TARGET] == 0).sum()):,}")
    print("Missing values remaining:",
          f"{int(cleaned.isna().sum().sum()):,}")
    print("\nCleaned dataset:", output_path)
    print("Cleaning audit:", audit_path)
    print("\nCleaning completed successfully.")


if __name__ == "__main__":
    main()
