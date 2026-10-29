
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "creditcard.csv"
FIGURES_DIR = ROOT / "outputs" / "figures"
REPORTS_DIR = ROOT / "outputs" / "reports"

sns.set_theme(style="whitegrid")


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading dataset for EDA...")
    df = pd.read_csv(DATA_PATH)

    # Basic validation
    required = {"Time", "Amount", "Class"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"Missing required columns: {required - set(df.columns)}"
        )

    fraud = df[df["Class"] == 1]
    genuine = df[df["Class"] == 0]

    total = len(df)
    fraud_count = len(fraud)
    genuine_count = len(genuine)
    fraud_rate = fraud_count / total * 100

    duplicate_count = int(df.duplicated().sum())

    # Investigate duplicate rows without deleting them
    duplicate_rows = df[df.duplicated(keep=False)]
    duplicate_class_counts = (
        duplicate_rows["Class"].value_counts().to_dict()
    )

    fraud_amount = float(fraud["Amount"].sum())
    genuine_amount = float(genuine["Amount"].sum())

    report = [
        "FRAUDSHIELD AI - EDA REPORT",
        "=" * 45,
        f"Total transactions: {total:,}",
        f"Genuine transactions: {genuine_count:,}",
        f"Fraudulent transactions: {fraud_count:,}",
        f"Fraud percentage: {fraud_rate:.4f}%",
        f"Missing values: {int(df.isna().sum().sum()):,}",
        f"Exact duplicate rows after first occurrence: {duplicate_count:,}",
        f"Class counts among all rows belonging to duplicate groups: "
        f"{duplicate_class_counts}",
        "",
        f"Total amount in fraud-labelled rows: {fraud_amount:,.2f}",
        f"Total amount in genuine-labelled rows: {genuine_amount:,.2f}",
        f"Mean transaction amount: {df['Amount'].mean():,.2f}",
        f"Median transaction amount: {df['Amount'].median():,.2f}",
        f"Mean fraud-labelled transaction amount: "
        f"{fraud['Amount'].mean():,.2f}",
        f"Mean genuine transaction amount: "
        f"{genuine['Amount'].mean():,.2f}",
        "",
        "Note: Fraud-labelled amounts are not necessarily confirmed",
        "financial losses or recovered amounts.",
        "Currency and original customer/merchant details are not supplied.",
    ]

    # Chart 1: Class distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(
        data=df,
        x="Class",
        hue="Class",
        order=[0, 1],
        legend=False,
        ax=ax,
    )
    ax.set_xticks([0, 1], ["Genuine (0)", "Fraud (1)"])
    ax.set_title("Transaction Class Distribution")
    ax.set_xlabel("Transaction class")
    ax.set_ylabel("Number of transactions")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f")
    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / "class_distribution.png", dpi=160
    )
    plt.close(fig)

    # Chart 2: Amount distribution, using a log axis
    # log1p allows zero-amount transactions to remain visible.
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, group in [
        ("Genuine", genuine),
        ("Fraud-labelled", fraud),
    ]:
        ax.hist(
            group["Amount"],
            bins=60,
            alpha=0.65,
            label=label,
            log=True,
        )
    ax.set_title("Transaction Amount Distribution")
    ax.set_xlabel("Transaction amount (dataset units)")
    ax.set_ylabel("Number of transactions (log scale)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / "amount_distribution.png", dpi=160
    )
    plt.close(fig)

    # Chart 3: Fraud rate across relative time buckets.
    # Time is elapsed seconds, not real clock time.
    time_df = df.copy()
    time_df["RelativeHour"] = (
        (time_df["Time"] // 3600).astype(int) % 24
    )

    hourly = (
        time_df.groupby("RelativeHour")
        .agg(
            transactions=("Class", "size"),
            frauds=("Class", "sum"),
        )
        .reset_index()
    )
    hourly["fraud_rate_pct"] = (
        hourly["frauds"] / hourly["transactions"] * 100
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(
        data=hourly,
        x="RelativeHour",
        y="fraud_rate_pct",
        marker="o",
        ax=ax,
    )
    ax.set_title("Fraud Rate by Relative Hour")
    ax.set_xlabel("Hour bucket relative to dataset start")
    ax.set_ylabel("Fraud-labelled transactions (%)")
    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / "fraud_rate_by_relative_hour.png",
        dpi=160,
    )
    plt.close(fig)

    # Save a compact hourly summary
    hourly.to_csv(
        REPORTS_DIR / "fraud_rate_by_relative_hour.csv",
        index=False,
    )

    # Write report
    report_path = REPORTS_DIR / "eda_summary.txt"
    report_path.write_text("\n".join(report), encoding="utf-8")

    print("\n".join(report))
    print("\nCharts saved to:", FIGURES_DIR)
    print("Report saved to:", report_path)
    print("\nEDA completed successfully.")


if __name__ == "__main__":
    main()
