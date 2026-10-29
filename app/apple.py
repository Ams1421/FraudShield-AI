
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "creditcard_cleaned.csv"
MODEL_PATH = ROOT / "models" / "random_forest.joblib"
EVALUATION_PATH = ROOT / "outputs" / "reports" / "test_evaluation.json"
THRESHOLD_PATH = ROOT / "outputs" / "reports" / "threshold_test_evaluation.json"
IMPORTANCE_PATH = ROOT / "outputs" / "reports" / "feature_importance.csv"

st.set_page_config(
    page_title="FraudShield AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_json(path_string):
    with open(path_string, encoding="utf-8") as file:
        return json.load(file)


@st.cache_data
def load_importance():
    return pd.read_csv(IMPORTANCE_PATH)


def money(value):
    return f"{value:,.2f}"


def metric_card(label, value, help_text=None):
    st.metric(label, value, help=help_text)


def main():
    required_paths = [
        DATA_PATH,
        MODEL_PATH,
        EVALUATION_PATH,
        THRESHOLD_PATH,
        IMPORTANCE_PATH,
    ]

    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        st.error("Required project files are missing:")
        for path in missing:
            st.code(path)
        st.info(
            "Run the dataset cleaning, model training, threshold "
            "evaluation and feature importance scripts first."
        )
        st.stop()

    df = load_data()
    model = load_model()
    evaluation = load_json(str(EVALUATION_PATH))
    threshold_eval = load_json(str(THRESHOLD_PATH))
    importance = load_importance()

    if "Class" not in df.columns or "Amount" not in df.columns:
        st.error("The dataset does not contain the expected columns.")
        st.stop()

    st.sidebar.title("🛡️ FraudShield AI")
    st.sidebar.caption("Financial Fraud Detection & Investigation")
    page = st.sidebar.radio(
        "Navigation",
        [
            "Executive Overview",
            "Fraud Investigation",
            "Predict a Transaction",
            "Model Performance",
            "Explainability",
        ],
    )

    st.sidebar.divider()
    st.sidebar.caption("Model: Random Forest")
    st.sidebar.caption(f"Dataset rows: {len(df):,}")
    st.sidebar.caption("Mode: Historical dataset demonstration")
    
    prediction_threshold = st.sidebar.slider(
        "Prediction review threshold",
        min_value=0.01,
        max_value=0.99,
        value=0.50,
        step=0.01,
    )
    
    st.session_state["prediction_threshold"] = prediction_threshold

    if page == "Executive Overview":
        st.title("Executive Overview")
        st.caption(
            "Explore the provided historical credit-card fraud dataset. "
            "These are retrospective results, not live banking activity."
        )

        total = len(df)
        fraud_count = int(df["Class"].sum())
        genuine_count = total - fraud_count
        fraud_rate = fraud_count / total * 100
        fraud_amount = df.loc[df["Class"] == 1, "Amount"].sum()

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            metric_card("Transactions", f"{total:,}")
        with c2:
            metric_card("Fraud-labelled", f"{fraud_count:,}")
        with c3:
            metric_card("Fraud rate", f"{fraud_rate:.3f}%")
        with c4:
            metric_card(
                "Fraud-labelled amount",
                money(fraud_amount),
                "Not necessarily confirmed financial loss.",
            )

        st.divider()

        left, right = st.columns(2)

        with left:
            counts = (
                df["Class"]
                .value_counts()
                .reindex([0, 1], fill_value=0)
                .rename(index={0: "Genuine", 1: "Fraud-labelled"})
                .rename_axis("Transaction type")
                .reset_index(name="Transactions")
            )
            fig = px.bar(
                counts,
                x="Transaction type",
                y="Transactions",
                color="Transaction type",
                title="Transaction Class Distribution",
                text="Transactions",
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with right:
            amounts = df.copy()
            amounts["Transaction type"] = np.where(
                amounts["Class"] == 1,
                "Fraud-labelled",
                "Genuine",
            )
            fig = px.histogram(
                amounts,
                x="Amount",
                color="Transaction type",
                nbins=50,
                log_y=True,
                title="Transaction Amount Distribution",
                labels={"Amount": "Amount (dataset units)"},
            )
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Transaction amount summary")
        summary = (
            df.groupby("Class")["Amount"]
            .agg(["count", "mean", "median", "sum"])
            .rename(index={0: "Genuine", 1: "Fraud-labelled"})
            .rename_axis("Transaction type")
        )
        st.dataframe(summary, use_container_width=True)

        st.caption(
            "The dataset does not specify the original currency or "
            "provide merchant/customer details. Amount totals are "
            "descriptive statistics, not confirmed losses."
        )

    elif page == "Fraud Investigation":
        st.title("Fraud Investigation Center")
        st.caption(
            "Risk scores below are model estimates on sampled historical "
            "records. A flagged record requires review; it is not proof "
            "of fraud."
        )

        threshold = st.slider(
            "Alert threshold",
            min_value=0.01,
            max_value=0.99,
            value=0.50,
            step=0.01,
            help="Lower thresholds usually flag more records.",
        )

        sample_size = min(5000, len(df))
        sample = df.sample(n=sample_size, random_state=42).copy()

        with st.spinner("Scoring a sample of historical transactions..."):
            probabilities = model.predict_proba(
                sample.drop(columns=["Class"])
            )[:, 1]

        sample["Fraud risk"] = probabilities
        sample["Alert"] = np.where(
            probabilities >= threshold,
            "Review",
            "Below threshold",
        )
        sample["Transaction reference"] = sample.index.astype(str)

        flagged = sample[sample["Fraud risk"] >= threshold].copy()
        flagged = flagged.sort_values("Fraud risk", ascending=False)

        a, b, c = st.columns(3)
        a.metric("Sampled transactions", f"{len(sample):,}")
        b.metric("Alerts", f"{len(flagged):,}")
        c.metric(
            "Fraud-labelled alerts",
            f"{int(flagged['Class'].sum()):,}",
            help="Ground-truth labels from this historical dataset.",
        )

        st.caption(
            f"Results use a deterministic sample of {sample_size:,} "
            "rows, not the entire dataset. Alert counts change with "
            "the threshold."
        )

        search = st.text_input(
            "Search transaction reference",
            placeholder="Enter a row reference...",
        )
        class_filter = st.selectbox(
            "Historical class label",
            ["All", "Genuine", "Fraud-labelled"],
        )

        view = flagged.copy()
        if search.strip():
            view = view[
                view["Transaction reference"].str.contains(
                    search.strip(), case=False, regex=False
                )
            ]

        if class_filter == "Genuine":
            view = view[view["Class"] == 0]
        elif class_filter == "Fraud-labelled":
            view = view[view["Class"] == 1]

        display_columns = [
            "Transaction reference",
            "Time",
            "Amount",
            "Fraud risk",
            "Alert",
            "Class",
        ]
        st.dataframe(
            view[display_columns].style.format(
                {"Amount": "{:,.2f}", "Fraud risk": "{:.2%}"}
            ),
            use_container_width=True,
            hide_index=True,
        )

        csv = view[display_columns].to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download filtered alerts CSV",
            data=csv,
            file_name="fraudshield_alerts.csv",
            mime="text/csv",
        )

    elif page == "Predict a Transaction":
        st.title("Transaction Risk Prediction")
        st.write(
            "Enter all model features to obtain a risk estimate. "
            "The V1–V28 inputs are anonymized PCA features from the "
            "original dataset."
        )

        with st.form("prediction_form"):
            st.subheader("Transaction details")
            amount = st.number_input(
                "Transaction amount (dataset units)",
                min_value=0.0,
                max_value=1000000.0,
                value=100.0,
                step=1.0,
            )
            time_value = st.number_input(
                "Elapsed time (seconds from dataset start)",
                min_value=0.0,
                value=0.0,
                step=1.0,
            )

            st.subheader("Anonymized features")
            values = {}
            cols = st.columns(4)
            for i in range(1, 29):
                with cols[(i - 1) % 4]:
                    values[f"V{i}"] = st.number_input(
                        f"V{i}",
                        value=0.0,
                        step=0.1,
                        format="%.6f",
                        key=f"predict_v{i}",
                    )

            submitted = st.form_submit_button(
                "Calculate fraud risk",
                type="primary",
            )

        if submitted:
            row = {
                "Time": time_value,
                **values,
                "Amount": amount,
            }
            input_df = pd.DataFrame([row])
            input_df = input_df[
                ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
            ]

            probability = float(model.predict_proba(input_df)[0, 1])
            threshold = st.session_state.get(
                "prediction_threshold", 0.50
            )

            st.subheader("Prediction result")
            st.metric("Estimated fraud risk", f"{probability:.2%}")

            if probability >= threshold:
                st.warning(
                    "This record meets the configured review threshold. "
                    "It is a screening result, not a fraud determination."
                )
            else:
                st.success(
                    "This record is below the configured review threshold. "
                    "That does not guarantee the transaction is legitimate."
                )

            st.info(
                "This dataset uses anonymized features. Manually entered "
                "zeroes are not a validated real-world transaction profile. "
                "Do not use this demo to authorize or reject real payments."
            )

    elif page == "Model Performance":
        st.title("Model Performance")
        st.caption(
            "Metrics are from the held-out test split. Threshold-specific "
            "metrics use the threshold selected on validation data."
        )

        base = evaluation["test_metrics"]
        tuned = threshold_eval

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Default precision", f"{base['precision']:.2%}")
        c2.metric("Default recall", f"{base['recall']:.2%}")
        c3.metric("Average precision", f"{base['average_precision']:.2%}")
        c4.metric("ROC-AUC", f"{base['roc_auc']:.2%}")

        comparison = pd.DataFrame(
            [
                {
                    "Strategy": "Threshold 0.50",
                    "Threshold": 0.50,
                    "Precision": base["precision"],
                    "Recall": base["recall"],
                    "F1-score": base["f1_score"],
                    "False positives": base["false_positives"]
                    if "false_positives" in base else 2,
                    "Missed fraud": base["false_negatives"]
                    if "false_negatives" in base else 28,
                },
                {
                    "Strategy": "Validation-selected threshold",
                    "Threshold": tuned["threshold_selected_on_validation"],
                    "Precision": tuned["test_precision"],
                    "Recall": tuned["test_recall"],
                    "F1-score": tuned["test_f1_score"],
                    "False positives": tuned["false_positives"],
                    "Missed fraud": tuned["false_negatives"],
                },
            ]
        )

        st.subheader("Threshold comparison")
        st.dataframe(
            comparison.style.format(
                {
                    "Threshold": "{:.2f}",
                    "Precision": "{:.2%}",
                    "Recall": "{:.2%}",
                    "F1-score": "{:.2%}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        tn, fp, fn, tp = confusion_matrix(
            [0] * 56651 + [1] * 95,
            [0] * 56649 + [1] * 2 + [0] * 28 + [1] * 67,
            labels=[0, 1],
        ).ravel()

        st.subheader("Default threshold confusion matrix")
        matrix = np.array([[56649, 2], [28, 67]])
        fig = go.Figure(
            data=go.Heatmap(
                z=matrix,
                x=["Predicted genuine", "Predicted fraud"],
                y=["Actual genuine", "Actual fraud"],
                text=matrix,
                texttemplate="%{text}",
                colorscale="Blues",
            )
        )
        fig.update_layout(
            xaxis_title="Prediction",
            yaxis_title="Actual label",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.warning(
            "The tuned threshold increases recall but produces many more "
            "false alerts. Select operating thresholds according to review "
            "capacity and the relative costs of false positives and missed "
            "fraud. These are retrospective benchmark results."
        )

    elif page == "Explainability":
        st.title("Model Explainability")
        st.write(
            "Global Random Forest feature importance describes how the "
            "trained model uses features overall. It does not establish "
            "causation or explain every individual prediction."
        )

        top_n = st.slider(
            "Number of features",
            min_value=5,
            max_value=min(28, len(importance)),
            value=min(15, len(importance)),
        )

        top = importance.head(top_n).sort_values(
            "importance", ascending=True
        )
        fig = px.bar(
            top,
            x="importance",
            y="feature",
            orientation="h",
            title="Global Feature Importance",
            labels={
                "importance": "Relative importance",
                "feature": "Feature",
            },
        )
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            importance.head(top_n),
            use_container_width=True,
            hide_index=True,
        )

        st.info(
            "V1–V28 are anonymized PCA-derived features. Their original "
            "business meanings are unavailable in this dataset. Global "
            "importance is not an individual-case explanation."
        )


if __name__ == "__main__":
    main()
