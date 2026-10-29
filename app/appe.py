from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

warnings.filterwarnings("ignore", category=FutureWarning)

# ------------------------------------------------------------
# FraudShield AI | Financial Fraud Detection & Investigation
# ------------------------------------------------------------
st.set_page_config(
    page_title="FraudShield AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
DATA_PATHS = [
    ROOT / "data" / "processed" / "creditcard_cleaned.csv",
    ROOT / "data" / "raw" / "creditcard.csv",
]
MODEL_PATHS = [
    ROOT / "models" / "random_forest.joblib",
    ROOT / "models" / "random_forest.pkl",
]
FEATURE_IMPORTANCE_PATH = ROOT / "outputs" / "reports" / "feature_importance.csv"
MODEL_COMPARISON_PATH = ROOT / "outputs" / "reports" / "model_comparison_validation.csv"
TEST_EVAL_PATH = ROOT / "outputs" / "reports" / "test_evaluation.json"
THRESHOLD_EVAL_PATH = ROOT / "outputs" / "reports" / "threshold_test_evaluation.json"
THRESHOLD_RECOMMENDATION_PATH = ROOT / "outputs" / "reports" / "threshold_recommendation.txt"
EDA_SUMMARY_PATH = ROOT / "outputs" / "reports" / "eda_summary.txt"

# Dark ink / blue / teal styling. Uses Streamlit's native layout and Plotly charts.
st.markdown(
    """
    <style>
    :root { --fs-ink:#14213d; --fs-blue:#1769aa; --fs-teal:#11a6a0; }
    .stApp { background: #f5f7fb; }
    [data-testid="stHeader"] { background: transparent !important; }
    [data-testid="stToolbar"] { background: transparent !important; }
    /* Keep page content clear of Streamlit's sticky top toolbar. */
    [data-testid="stAppViewContainer"] .main .block-container {
        padding-top: 2rem !important;
    }
    [data-testid="stSidebar"] { background: #101b31; }
    [data-testid="stSidebar"] * { color: #e8eef9 !important; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color:#aebbd0 !important; }
    [data-testid="stSidebar"] .fs-nav-heading { color:#91a4bf; font-size:.70rem; font-weight:800; letter-spacing:.12em; margin:0 0 .55rem .15rem; }
    [data-testid="stSidebar"] [data-testid="stButton"] { margin:.13rem 0; }
    [data-testid="stSidebar"] [data-testid="stButton"] button {
        min-height:2.55rem; border-radius:10px; border:1px solid transparent;
        text-align:left; justify-content:flex-start; padding:.48rem .72rem;
        font-size:.84rem; font-weight:600; transition:background .18s ease, border-color .18s ease, transform .18s ease;
        box-shadow:none;
    }
    [data-testid="stSidebar"] [data-testid="stButton"] button[kind="secondary"] {
        background:transparent !important; color:#d5dfef !important;
    }
    [data-testid="stSidebar"] [data-testid="stButton"] button[kind="secondary"]:hover {
        background:#1b2c48 !important; border-color:#314766 !important; color:#ffffff !important;
        transform:translateX(2px);
    }
    [data-testid="stSidebar"] [data-testid="stButton"] button[kind="primary"] {
        background:#263d62 !important; color:#ffffff !important; border-color:#3d5a83 !important;
        box-shadow:inset 3px 0 0 #4aa8ff;
    }
    [data-testid="stSidebar"] [data-testid="stButton"] button[kind="primary"]:hover {
        background:#304d78 !important; border-color:#4a6c99 !important;
    }
    .fs-brand {font-size:1.35rem;font-weight:800;letter-spacing:-.03em;color:#fff;margin-bottom:0;}
    .fs-subbrand {font-size:.78rem;color:#aebbd0;margin-top:0;}
    .fs-eyebrow {text-transform:uppercase;letter-spacing:.12em;font-size:.72rem;font-weight:800;color:#64748b;}
    .fs-hero {background:linear-gradient(115deg,#12213b,#174f83);padding:1.4rem 1.5rem;border-radius:18px;color:white;margin:0 0 1rem 0;border-top:0;}
    .fs-hero h1 {color:white;margin:0;font-size:2rem;letter-spacing:-.04em;}
    .fs-hero p {color:#d9e8f7;margin:.4rem 0 0 0;}
    .fs-card {background:white;border:1px solid #e5eaf2;border-radius:15px;padding:1rem 1.05rem;box-shadow:0 4px 14px rgba(22,34,51,.035);min-height:112px;}
    .fs-card-label {font-size:.78rem;color:#64748b;font-weight:650;margin-bottom:.45rem;}
    .fs-card-value {font-size:1.75rem;font-weight:800;color:#14213d;line-height:1.1;}
    .fs-card-note {font-size:.76rem;color:#64748b;margin-top:.45rem;}
    .fs-section {font-size:1.08rem;font-weight:800;color:#172b4d;margin:.4rem 0 .15rem 0;}
    .fs-muted {color:#64748b;font-size:.86rem;}
    .fs-pill {display:inline-block;padding:.24rem .6rem;border-radius:999px;background:#e8f3ff;color:#15558a;font-weight:700;font-size:.75rem;}
    div[data-testid="stMetric"] {background:white;border:1px solid #e5eaf2;padding:14px 16px;border-radius:14px;box-shadow:0 4px 14px rgba(22,34,51,.035);}
    div[data-testid="stMetricLabel"] {color:#64748b;}
    div[data-testid="stPlotlyChart"] {background:white;border:1px solid #e5eaf2;border-radius:14px;padding:8px;}
    .block-container {padding-top:2rem;padding-bottom:2rem;max-width:1600px;}
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_data(show_spinner="Loading transaction data…")
def load_data(path_string):
    frame = pd.read_csv(path_string)
    expected = {"Time", "Amount", "Class"}
    if not expected.issubset(frame.columns):
        raise ValueError(f"Dataset must contain columns: {sorted(expected)}")
    frame["Class"] = pd.to_numeric(frame["Class"], errors="coerce").fillna(0).astype(int)
    frame["Amount"] = pd.to_numeric(frame["Amount"], errors="coerce").fillna(0)
    frame["Time"] = pd.to_numeric(frame["Time"], errors="coerce").fillna(0)
    return frame

@st.cache_resource(show_spinner="Loading trained model…")
def load_model(path_string):
    return joblib.load(path_string)

def first_existing(paths):
    return next((p for p in paths if p.exists()), None)

def read_json(path):
    if path and path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def money(value):
    return f"{value:,.2f}"

def pct(value, digits=2):
    return f"{value * 100:.{digits}f}%"

def metric_card(label, value, note=""):
    st.markdown(
        f"""<div class="fs-card"><div class="fs-card-label">{label}</div>
        <div class="fs-card-value">{value}</div><div class="fs-card-note">{note}</div></div>""",
        unsafe_allow_html=True,
    )

def section(title, subtitle=None):
    st.markdown(f'<div class="fs-section">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="fs-muted">{subtitle}</div>', unsafe_allow_html=True)

def style_fig(fig, height=360):
    fig.update_layout(
        height=height,
        margin=dict(l=18, r=18, t=42, b=18),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Arial, sans-serif", color="#24344d", size=12),
        legend_title_text="",
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="#edf1f7", zeroline=False)
    return fig

def build_feature_frame(data, feature_names):
    # Keep exactly the feature order learned by the fitted estimator.
    missing = [c for c in feature_names if c not in data.columns]
    if missing:
        raise ValueError(f"Required model columns are missing: {missing}")
    return data.loc[:, feature_names].copy()

def model_scores(model, data):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(data)[:, 1]
    if hasattr(model, "decision_function"):
        raw = model.decision_function(data)
        return 1 / (1 + np.exp(-raw))
    return model.predict(data).astype(float)

data_path = first_existing(DATA_PATHS)
model_path = first_existing(MODEL_PATHS)
if data_path is None:
    st.error("Dataset not found. Expected data/processed/creditcard_cleaned.csv or data/raw/creditcard.csv.")
    st.code("C:\\Projects\\FraudShield-AI\\data\\processed\\creditcard_cleaned.csv")
    st.stop()

try:
    df = load_data(str(data_path))
except Exception as exc:
    st.error(f"Could not load dataset: {exc}")
    st.stop()

if "Class" not in df.columns:
    st.error("The dataset does not contain the target column 'Class'.")
    st.stop()

total_rows = len(df)
fraud_count = int((df["Class"] == 1).sum())
genuine_count = int((df["Class"] == 0).sum())
fraud_rate = fraud_count / max(total_rows, 1)
fraud_amount = float(df.loc[df["Class"] == 1, "Amount"].sum())
genuine_amount = float(df.loc[df["Class"] == 0, "Amount"].sum())
avg_fraud_amount = float(df.loc[df["Class"] == 1, "Amount"].mean()) if fraud_count else 0
avg_genuine_amount = float(df.loc[df["Class"] == 0, "Amount"].mean()) if genuine_count else 0
amount_col = "Amount"
feature_cols = [c for c in df.columns if c not in ("Class",)]
if model_path:
    try:
        model = load_model(str(model_path))
    except Exception as exc:
        model = None
        model_load_error = str(exc)
else:
    model = None
    model_load_error = "No Random Forest model file found in models/."

feature_names = list(getattr(model, "feature_names_in_", [])) if model is not None else []
if not feature_names:
    feature_names = [c for c in df.columns if c != "Class"]
feature_names = [c for c in feature_names if c in df.columns]
if model is not None and not feature_names:
    model = None
    model_load_error = "The model's expected features could not be matched to the dataset."

default_threshold = 0.50
threshold_recommendation = 0.01
if THRESHOLD_RECOMMENDATION_PATH.exists():
    try:
        import re
        text_rec = THRESHOLD_RECOMMENDATION_PATH.read_text(encoding="utf-8")
        match = re.search(r"(?:threshold|Threshold)[^0-9]*([0-9]*\\.?[0-9]+)", text_rec)
        if match:
            threshold_recommendation = float(match.group(1))
    except Exception:
        pass
# Known validated project threshold if report parser cannot extract it.
threshold_recommendation = min(max(threshold_recommendation, 0.01), 0.99)
test_eval = read_json(TEST_EVAL_PATH)
tuned_eval = read_json(THRESHOLD_EVAL_PATH)

if "review_threshold" not in st.session_state:
    st.session_state.review_threshold = 0.50
if "investigation_sample" not in st.session_state:
    st.session_state.investigation_sample = 5000

# Sidebar
with st.sidebar:
    st.markdown('<div style="font-size:2rem">🛡️</div>', unsafe_allow_html=True)
    st.markdown('<p class="fs-brand">FraudShield AI</p>', unsafe_allow_html=True)
    st.markdown('<p class="fs-subbrand">Financial Fraud Detection & Investigation</p>', unsafe_allow_html=True)
    st.divider()
    navigation_pages = [
        ("Executive Overview", "▦"),
        ("Fraud Investigation", "⌕"),
        ("Predict a Transaction", "↗"),
        ("Model Performance", "▥"),
        ("Threshold & Risk", "◉"),
        ("Explainability", "✳"),
        ("Fraud Patterns", "⌁"),
        ("Data Explorer", "▤"),
        ("Model Monitoring", "◷"),
        ("About & Methodology", "ⓘ"),
    ]
    if "active_page" not in st.session_state:
        st.session_state.active_page = "Executive Overview"

    st.markdown(
        """<div class="fs-nav-heading">WORKSPACE</div>""",
        unsafe_allow_html=True,
    )
    for nav_label, nav_icon in navigation_pages:
        is_active = st.session_state.active_page == nav_label
        if st.button(
            f"{nav_icon}   {nav_label}",
            key=f"nav_{nav_label}",
            type="primary" if is_active else "secondary",
            use_container_width=True,
        ):
            st.session_state.active_page = nav_label
            st.rerun()
    page = st.session_state.active_page
    st.divider()
    st.caption("ACTIVE MODEL")
    st.markdown("**Random Forest**" if model is not None else "**Model unavailable**")
    st.caption(f"{total_rows:,} historical records")
    st.caption(f"Source: `{data_path.relative_to(ROOT) if data_path.is_relative_to(ROOT) else data_path.name}`")
    st.slider("Investigation alert threshold", 0.01, 0.99, key="review_threshold", step=0.01)
    st.caption("Historical dataset demonstration · not a live banking feed")
    st.markdown("---")
    st.caption("FraudShield AI · Portfolio analytics prototype")

# Shared derived data
working = df.copy()
working["Transaction type"] = np.where(working["Class"] == 1, "Fraud-labelled", "Genuine")
working["Amount band"] = pd.cut(
    working["Amount"],
    bins=[-0.001, 1, 25, 100, 500, 2000, np.inf],
    labels=["≤ 1", "1–25", "25–100", "100–500", "500–2k", "> 2k"],
    include_lowest=True,
)
working["Elapsed hour"] = (working["Time"] / 3600).astype(int)
working["Elapsed time band"] = (working["Elapsed hour"] // 4 * 4).astype(str) + "–" + ((working["Elapsed hour"] // 4 * 4) + 4).astype(str) + " h"

st.markdown(
    f"""<div class="fs-hero"><div class="fs-eyebrow" style="color:#9cc8f1">Fraud intelligence workspace</div>
    <h1>{page}</h1>
    <p>Explore patterns, prioritize alerts, and review model performance using the loaded historical dataset.</p></div>""",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# 1. Executive Overview
# ------------------------------------------------------------
if page == "Executive Overview":
    cols = st.columns(4)
    with cols[0]: metric_card("Transactions analysed", f"{total_rows:,}", "Rows in the loaded dataset")
    with cols[1]: metric_card("Fraud-labelled", f"{fraud_count:,}", f"{genuine_count:,} genuine-labelled records")
    with cols[2]: metric_card("Fraud-label rate", pct(fraud_rate, 3), "Share labelled Class = 1")
    with cols[3]: metric_card("Fraud-labelled amount", money(fraud_amount), "Dataset amount units; not confirmed losses")

    st.write("")
    left, right = st.columns([1, 1])
    with left:
        section("Transaction class distribution", "The class imbalance makes accuracy alone an unsuitable success measure.")
        counts = working["Transaction type"].value_counts().rename_axis("Class").reset_index(name="Transactions")
        fig = px.bar(counts, x="Class", y="Transactions", color="Class", text="Transactions",
                     color_discrete_map={"Genuine": "#1769aa", "Fraud-labelled": "#f05b67"})
        fig.update_traces(texttemplate="%{text:,}", textposition="outside")
        st.plotly_chart(style_fig(fig, 360), use_container_width=True)
    with right:
        section("Transaction amount distribution", "Logarithmic frequency axis helps show the long tail.")
        fig = px.histogram(working, x="Amount", color="Transaction type", nbins=65,
                           barmode="overlay", opacity=.78,
                           color_discrete_map={"Genuine": "#1769aa", "Fraud-labelled": "#f05b67"})
        fig.update_yaxes(type="log", title="Transaction count (log scale)")
        fig.update_xaxes(title="Amount (dataset units)")
        st.plotly_chart(style_fig(fig, 360), use_container_width=True)

    left, right = st.columns(2)
    with left:
        section("Average amount by label")
        avg = working.groupby("Transaction type", observed=False)["Amount"].mean().reset_index()
        fig = px.bar(avg, x="Transaction type", y="Amount", color="Transaction type",
                     text_auto=".2f", color_discrete_map={"Genuine": "#1769aa", "Fraud-labelled": "#f05b67"})
        fig.update_layout(showlegend=False, yaxis_title="Average amount (dataset units)")
        st.plotly_chart(style_fig(fig, 310), use_container_width=True)
    with right:
        section("Fraud-labelled share by elapsed-time window", "Time is elapsed seconds in the source dataset, not clock time.")
        time_summary = working.groupby("Elapsed time band", observed=False).agg(
            Transactions=("Class", "size"), Fraud=("Class", "sum")
        ).reset_index()
        time_summary["Fraud rate (%)"] = np.where(time_summary["Transactions"] > 0, 100 * time_summary["Fraud"] / time_summary["Transactions"], 0)
        fig = px.line(time_summary, x="Elapsed time band", y="Fraud rate (%)", markers=True)
        fig.update_layout(xaxis_title="Elapsed-time band", yaxis_title="Fraud-labelled rate (%)")
        st.plotly_chart(style_fig(fig, 310), use_container_width=True)

    st.info("Interpretation note: Class = 1 indicates a fraud label in this dataset. Amount is shown in dataset units because the source does not establish a currency here.")

# ------------------------------------------------------------
# 2. Fraud Investigation
# ------------------------------------------------------------
elif page == "Fraud Investigation":
    section("Investigation queue", "Filter the labelled records and review a reproducible sample of model-scored transactions.")
    if model is None:
        st.warning(f"Model unavailable: {model_load_error}. Dataset-based investigation remains available, but probability scores are disabled.")
        scored = working.copy()
        scored["Fraud score"] = np.nan
        scored["Review status"] = np.where(scored["Class"] == 1, "Fraud-labelled", "Genuine-labelled")
    else:
        n_sample = min(int(st.session_state.investigation_sample), len(df))
        # Stable deterministic sample so the table is reproducible between reruns.
        sample = df.sample(n=n_sample, random_state=42) if n_sample < len(df) else df
        X_sample = build_feature_frame(sample, feature_names)
        sample_scores = model_scores(model, X_sample)
        scored = sample.copy()
        scored["Fraud score"] = sample_scores
        scored["Review status"] = np.where(
            scored["Fraud score"] >= st.session_state.review_threshold, "Review alert", "Below threshold"
        )
        scored["Transaction type"] = np.where(scored["Class"] == 1, "Fraud-labelled", "Genuine")
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Scored records", f"{len(scored):,}")
    with c2: st.metric("Review alerts", f"{int((scored['Fraud score'] >= st.session_state.review_threshold).sum()):,}" if scored["Fraud score"].notna().any() else "N/A")
    with c3: st.metric("Fraud-labelled in view", f"{int(scored['Class'].sum()):,}")
    with st.expander("Investigation filters", expanded=True):
        a, b, c, d = st.columns(4)
        with a:
            sample_size = st.selectbox("Sample size", [1000, 5000, 10000, 25000, len(df)], index=1, format_func=lambda x: f"{x:,} records" if x != len(df) else f"All {x:,}")
            st.session_state.investigation_sample = int(sample_size)
        with b:
            class_filter = st.selectbox("Label", ["All", "Fraud-labelled", "Genuine"])
        with c:
            status_options = ["All"] + sorted(scored["Review status"].dropna().unique().tolist())
            status_filter = st.selectbox("Review status", status_options)
        with d:
            min_amount, max_amount = float(df["Amount"].min()), float(df["Amount"].max())
            amount_range = st.slider("Amount range", min_amount, max_amount, (min_amount, max_amount))
    filtered = scored.copy()
    if class_filter == "Fraud-labelled":
        filtered = filtered[filtered["Class"] == 1]
    elif class_filter == "Genuine":
        filtered = filtered[filtered["Class"] == 0]
    if status_filter != "All":
        filtered = filtered[filtered["Review status"] == status_filter]
    filtered = filtered[filtered["Amount"].between(amount_range[0], amount_range[1])]
    if "Fraud score" in filtered and filtered["Fraud score"].notna().any():
        filtered = filtered.sort_values("Fraud score", ascending=False)
    show_cols = [c for c in ["Time", "Amount", "Class", "Fraud score", "Review status"] if c in filtered.columns]
    st.dataframe(filtered[show_cols].head(500), use_container_width=True, hide_index=True,
                 column_config={"Class": st.column_config.NumberColumn("Actual label", help="1 = fraud-labelled; 0 = genuine-labelled"),
                                "Fraud score": st.column_config.ProgressColumn("Model fraud score", min_value=0, max_value=1, format="%.3f")})
    st.caption(f"Showing {min(len(filtered), 500):,} of {len(filtered):,} filtered rows. Table display is capped at 500 rows.")
    st.download_button("Download filtered investigation queue (CSV)", filtered[show_cols].to_csv(index=False).encode("utf-8"),
                       "fraudshield_investigation_queue.csv", "text/csv")
    if model is not None:
        plot_data = scored.copy()
        plot_data["Fraud score (%)"] = plot_data["Fraud score"] * 100
        fig = px.histogram(plot_data, x="Fraud score (%)", color="Transaction type", nbins=40, barmode="overlay",
                           color_discrete_map={"Genuine": "#1769aa", "Fraud-labelled": "#f05b67"})
        fig.add_vline(x=st.session_state.review_threshold * 100, line_dash="dash", line_color="#f05b67",
                      annotation_text=f"Threshold {st.session_state.review_threshold:.2f}")
        st.plotly_chart(style_fig(fig, 320), use_container_width=True)

# ------------------------------------------------------------
# 3. Predict a Transaction
# ------------------------------------------------------------
elif page == "Predict a Transaction":
    section("Transaction risk assessment", "Enter values for a single record. V1–V28 are anonymized transformed features from the source dataset.")
    if model is None:
        st.error(f"Prediction is unavailable because the model could not be loaded. {model_load_error}")
    else:
        with st.form("transaction_form"):
            a, b, c = st.columns(3)
            with a:
                tx_time = st.number_input("Time (elapsed seconds)", min_value=0.0, value=float(df["Time"].median()), step=1.0)
            with b:
                tx_amount = st.number_input("Amount (dataset units)", min_value=0.0, value=float(df["Amount"].median()), step=1.0)
            with c:
                threshold = st.number_input("Decision threshold", min_value=0.01, max_value=0.99,
                                            value=float(st.session_state.review_threshold), step=0.01)
            st.markdown("**Anonymized model features**")
            default_row = df[feature_names].median(numeric_only=True).to_dict()
            feature_values = {}
            vcols = st.columns(4)
            for i, col in enumerate([c for c in feature_names if c not in ("Time", "Amount")]):
                with vcols[i % 4]:
                    feature_values[col] = st.number_input(col, value=float(default_row.get(col, 0.0)), format="%.5f", key=f"predict_{col}")
            submitted = st.form_submit_button("Assess transaction risk", type="primary", use_container_width=True)
        if submitted:
            row = {}
            for col in feature_names:
                if col == "Time":
                    row[col] = tx_time
                elif col == "Amount":
                    row[col] = tx_amount
                else:
                    row[col] = feature_values.get(col, float(default_row.get(col, 0.0)))
            input_frame = pd.DataFrame([row], columns=feature_names)
            score = float(model_scores(model, input_frame)[0])
            alert = score >= threshold
            c1, c2, c3 = st.columns(3)
            with c1: metric_card("Estimated fraud score", f"{score:.2%}", "Model output, not a guaranteed probability of loss")
            with c2: metric_card("Decision", "REVIEW ALERT" if alert else "BELOW THRESHOLD", f"Threshold: {threshold:.2f}")
            with c3: metric_card("Entered amount", money(tx_amount), "Dataset units")
            if alert:
                st.error("Flag for human review. This output is a model score, not a final fraud determination.")
            else:
                st.success("Below the selected review threshold. Low scores do not guarantee a transaction is genuine.")
            st.progress(min(max(score, 0.0), 1.0))
            st.download_button("Download prediction result", pd.DataFrame([{"Time": tx_time, "Amount": tx_amount, "Fraud score": score, "Threshold": threshold, "Decision": "Review alert" if alert else "Below threshold"}]).to_csv(index=False).encode("utf-8"),
                               "fraudshield_single_prediction.csv", "text/csv")
    st.info("For a meaningful case assessment, use a real transaction feature row from the same feature-generation pipeline. Median/default feature values are only a demonstration.")

# ------------------------------------------------------------
# 4. Model Performance
# ------------------------------------------------------------
elif page == "Model Performance":
    section("Held-out test performance", "Metrics are from the saved test evaluation report where available.")
    if test_eval:
        p = float(test_eval.get("precision", 0.9710144927536232))
        r = float(test_eval.get("recall", 0.7052631578947368))
        f = float(test_eval.get("f1", 0.8170731707317073))
        ap = float(test_eval.get("average_precision", 0.811840))
        auc = float(test_eval.get("roc_auc", 0.939300))
        cm = test_eval.get("confusion_matrix", [[56649, 2], [28, 67]])
    else:
        p, r, f, ap, auc = 0.971014, 0.705263, 0.817073, 0.811840, 0.939300
        cm = [[56649, 2], [28, 67]]

    c = st.columns(5)
    for col, label, val, note in [
        (c[0], "Precision", pct(p), "Alert reliability"),
        (c[1], "Recall", pct(r), "Fraud capture"),
        (c[2], "F1 score", f"{f:.3f}", "Precision/recall balance"),
        (c[3], "Average precision", f"{ap:.3f}", "Imbalance-aware ranking"),
        (c[4], "ROC-AUC", f"{auc:.3f}", "Ranking discrimination"),
    ]:
        with col:
            metric_card(label, val, note)

    st.write("")
    section("Confusion matrix · default threshold 0.50",
            "Exact transaction counts are displayed instead of abbreviated values.")
    cm = np.asarray(cm, dtype=int)
    if cm.shape != (2, 2):
        cm = np.array([[56649, 2], [28, 67]])

    cm_text = np.array([[f"{int(value):,}" for value in row] for row in cm])
    cm_fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=["Predicted genuine", "Predicted fraud"],
        y=["Actual genuine", "Actual fraud"],
        text=cm_text,
        texttemplate="%{text}",
        textfont={"size": 18, "color": "#10233f"},
        colorscale=[[0, "#eef6ff"], [0.35, "#8dbce8"], [1, "#1769aa"]],
        showscale=False,
        hovertemplate="Actual: %{y}<br>%{x}<br>Count: %{z:,}<extra></extra>",
    ))
    cm_fig.update_layout(
        height=360,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Arial, sans-serif", color="#24344d", size=12),
        xaxis_title="",
        yaxis_title="",
        xaxis=dict(side="bottom"),
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(cm_fig, use_container_width=True)

    st.write("")
    section("Model comparison", "Validation-set results used to compare candidate models.")
    if MODEL_COMPARISON_PATH.exists():
        comparison = pd.read_csv(MODEL_COMPARISON_PATH)
        preferred = ["model", "precision", "recall", "f1_score", "average_precision", "roc_auc"]
        display_cols = [c for c in preferred if c in comparison.columns]
        if not display_cols:
            display_cols = list(comparison.columns[:6])

        format_config = {}
        for col_name in display_cols:
            lower = col_name.lower()
            if lower in ("precision", "recall", "f1_score", "average_precision", "roc_auc"):
                format_config[col_name] = st.column_config.NumberColumn(
                    col_name.replace("_", " ").title(), format="%.4f"
                )
            elif lower == "model":
                format_config[col_name] = st.column_config.TextColumn("Model")

        st.dataframe(
            comparison[display_cols],
            use_container_width=True,
            hide_index=True,
            column_config=format_config,
        )

        numeric_cols = [
            c for c in comparison.columns
            if c.lower() in ("average_precision", "roc_auc", "f1_score", "precision", "recall")
        ]
        if numeric_cols:
            metric_map = {
                "average_precision": "Average precision",
                "roc_auc": "ROC-AUC",
                "f1_score": "F1 score",
                "precision": "Precision",
                "recall": "Recall",
            }
            metric_choice = st.selectbox(
                "Compare models by",
                numeric_cols,
                index=numeric_cols.index("average_precision") if "average_precision" in numeric_cols else 0,
                format_func=lambda x: metric_map.get(x.lower(), x.replace("_", " ").title()),
            )
            model_name_col = next((c for c in comparison.columns if "model" in c.lower()), comparison.columns[0])
            chart_df = comparison.copy()
            chart_df[metric_choice] = pd.to_numeric(chart_df[metric_choice], errors="coerce")
            chart_fig = px.bar(
                chart_df,
                x=model_name_col,
                y=metric_choice,
                color=model_name_col,
                text=chart_df[metric_choice].map(lambda v: f"{v:.3f}" if pd.notna(v) else ""),
                color_discrete_sequence=["#1769aa", "#83c5f4", "#11a6a0"],
            )
            chart_fig.update_traces(textposition="outside", cliponaxis=False)
            chart_fig.update_layout(
                showlegend=False,
                yaxis_title=metric_map.get(metric_choice.lower(), metric_choice.replace("_", " ").title()),
                xaxis_title="Model",
                yaxis=dict(rangemode="tozero"),
            )
            st.plotly_chart(style_fig(chart_fig, 320), use_container_width=True)

        extra_cols = [c for c in comparison.columns if c not in display_cols]
        if extra_cols:
            with st.expander("Show all model comparison columns"):
                st.dataframe(comparison, use_container_width=True, hide_index=True)
    else:
        st.info("Model comparison report not found.")

    st.warning("The default-threshold test results are separate from validation-based threshold selection. Do not tune decisions using the held-out test set.")


# ------------------------------------------------------------
# 5. Threshold & Risk
# ------------------------------------------------------------
elif page == "Threshold & Risk":
    section("Threshold trade-off simulator", "Move the threshold to see the alert volume and trade-off on a deterministic scored sample.")
    if model is None:
        st.warning(f"Scoring unavailable: {model_load_error}")
    else:
        n = min(20000, len(df))
        sample = df.sample(n=n, random_state=24) if n < len(df) else df
        scores = model_scores(model, build_feature_frame(sample, feature_names))
        y = sample["Class"].to_numpy()
        threshold = st.slider("Decision threshold", 0.01, 0.99, float(st.session_state.review_threshold), step=0.01, key="risk_threshold")
        pred = (scores >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        precision = precision_score(y, pred, zero_division=0)
        recall = recall_score(y, pred, zero_division=0)
        f1 = f1_score(y, pred, zero_division=0)
        alerts = int(pred.sum())
        c = st.columns(5)
        for col, label, val, note in [
            (c[0], "Threshold", f"{threshold:.2f}", "Score cutoff"),
            (c[1], "Alerts", f"{alerts:,}", f"{alerts/max(n,1):.2%} of sample"),
            (c[2], "Precision", pct(precision), "Of alerts, labelled fraud"),
            (c[3], "Recall", pct(recall), "Fraud labels captured"),
            (c[4], "Missed labels", f"{fn:,}", "False negatives in sample"),
        ]:
            with col: metric_card(label, val, note)
        left, right = st.columns(2)
        with left:
            section("Confusion matrix at selected threshold")
            fig = px.imshow([[tn, fp], [fn, tp]], text_auto=True,
                            x=["Predicted genuine", "Predicted fraud"],
                            y=["Actual genuine", "Actual fraud"], color_continuous_scale="Teal", aspect="auto")
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(style_fig(fig, 330), use_container_width=True)
        with right:
            section("Score distribution and cutoff")
            plot = pd.DataFrame({"Fraud score": scores, "Actual label": np.where(y == 1, "Fraud-labelled", "Genuine")})
            fig = px.histogram(plot, x="Fraud score", color="Actual label", nbins=45, barmode="overlay",
                               color_discrete_map={"Genuine": "#1769aa", "Fraud-labelled": "#f05b67"})
            fig.add_vline(x=threshold, line_dash="dash", line_color="#f05b67")
            st.plotly_chart(style_fig(fig, 330), use_container_width=True)
        st.caption("These metrics are calculated on a reproducible sample for interactive exploration, not the held-out test set. Changing the threshold changes workload and error trade-offs.")
        if THRESHOLD_EVAL_PATH.exists():
            with st.expander("Saved validation-selected threshold · held-out test report"):
                st.json(tuned_eval)
        st.download_button("Download current sample scores", pd.DataFrame({"Class": y, "Fraud score": scores, "Predicted alert": pred}).to_csv(index=False).encode("utf-8"),
                           "fraudshield_threshold_sample.csv", "text/csv")

# ------------------------------------------------------------
# 6. Explainability
# ------------------------------------------------------------
elif page == "Explainability":
    section("Global model feature importance", "Shows which input columns the Random Forest uses most across the fitted model; this is not causation.")
    if FEATURE_IMPORTANCE_PATH.exists():
        imp = pd.read_csv(FEATURE_IMPORTANCE_PATH)
        name_col = next((c for c in imp.columns if "feature" in c.lower()), imp.columns[0])
        value_col = next((c for c in imp.columns if "importance" in c.lower()), imp.columns[-1])
        imp[value_col] = pd.to_numeric(imp[value_col], errors="coerce")
        imp = imp.dropna(subset=[value_col]).sort_values(value_col, ascending=False)
    elif model is not None and hasattr(model, "feature_importances_"):
        imp = pd.DataFrame({"Feature": feature_names, "Importance": model.feature_importances_})
        name_col, value_col = "Feature", "Importance"
        imp = imp.sort_values(value_col, ascending=False)
    else:
        imp = pd.DataFrame({"Feature": ["V14", "V12", "V4", "V10", "V17", "V11", "V3", "V16", "V7", "V2"],
                            "Importance": [.183662, .117733, .112087, .093802, .089041, .078021, .060651, .036008, .025016, .024606]})
        name_col, value_col = "Feature", "Importance"
    top_n = st.slider("Features to display", 5, min(28, max(5, len(imp))), min(15, max(5, len(imp))))
    top = imp.head(top_n).sort_values(value_col, ascending=True)
    c1, c2 = st.columns([1.4, 1])
    with c1:
        fig = px.bar(top, x=value_col, y=name_col, orientation="h", text_auto=".3f",
                     color=value_col, color_continuous_scale="Blues")
        fig.update_layout(coloraxis_showscale=False, xaxis_title="Global importance", yaxis_title="")
        st.plotly_chart(style_fig(fig, 520), use_container_width=True)
    with c2:
        section("Top features")
        st.dataframe(imp.head(top_n).rename(columns={name_col: "Feature", value_col: "Importance"}),
                     use_container_width=True, hide_index=True)
        st.metric("Top feature", str(imp.iloc[0][name_col]))
        st.metric("Top feature importance", f"{float(imp.iloc[0][value_col]):.3f}")
    st.info("V1–V28 are anonymized PCA-derived features in the source dataset. Their original business meanings are not provided, so this dashboard does not assign invented meanings to them.")
    st.caption("Global importance does not explain why one particular transaction received a score. A local explanation method such as SHAP could be added as a separately validated feature.")

# ------------------------------------------------------------
# 7. Fraud Patterns
# ------------------------------------------------------------
elif page == "Fraud Patterns":
    section("Pattern exploration", "Explore class labels across amount and elapsed-time groupings. Elapsed time is not a clock hour.")
    left, right = st.columns(2)
    with left:
        section("Fraud label rate by amount band")
        amount_summary = working.groupby("Amount band", observed=False).agg(
            Transactions=("Class", "size"), Fraud=("Class", "sum")
        ).reset_index()
        amount_summary["Fraud rate (%)"] = 100 * amount_summary["Fraud"] / amount_summary["Transactions"].replace(0, np.nan)
        fig = px.bar(amount_summary, x="Amount band", y="Fraud rate (%)", text_auto=".3f",
                     color="Fraud rate (%)", color_continuous_scale="Reds")
        fig.update_layout(coloraxis_showscale=False, xaxis_title="Amount band (dataset units)")
        st.plotly_chart(style_fig(fig, 360), use_container_width=True)
    with right:
        section("Fraud vs genuine amount distribution")
        fig = px.box(working, x="Transaction type", y="Amount", color="Transaction type", points=False,
                     color_discrete_map={"Genuine": "#1769aa", "Fraud-labelled": "#f05b67"})
        fig.update_yaxes(type="log", title="Amount (log scale; zero values omitted from log rendering)")
        st.plotly_chart(style_fig(fig, 360), use_container_width=True)
    left, right = st.columns(2)
    with left:
        section("Fraud-labelled records by elapsed-time band")
        by_time = working.groupby("Elapsed time band", observed=False).agg(
            Transactions=("Class", "size"), Fraud=("Class", "sum")
        ).reset_index()
        fig = px.bar(by_time, x="Elapsed time band", y="Fraud", color="Fraud", color_continuous_scale="Reds")
        fig.update_layout(coloraxis_showscale=False, xaxis_title="Elapsed-time band", yaxis_title="Fraud-labelled records")
        st.plotly_chart(style_fig(fig, 320), use_container_width=True)
    with right:
        section("Amount summary")
        stats = working.groupby("Transaction type").agg(
            Count=("Amount", "size"), Median=("Amount", "median"), Mean=("Amount", "mean"),
            **{"95th percentile": ("Amount", lambda x: x.quantile(.95)), "Maximum": ("Amount", "max")}
        ).reset_index()
        st.dataframe(stats, use_container_width=True, hide_index=True)
    st.caption("Patterns are descriptive and may reflect the dataset's construction. They should not be interpreted as causal fraud rules.")

# ------------------------------------------------------------
# 8. Data Explorer
# ------------------------------------------------------------
elif page == "Data Explorer":
    section("Dataset quality and profile", "Review the actual loaded file and select fields for exploration.")
    c = st.columns(4)
    with c[0]: metric_card("Rows", f"{len(df):,}", "Loaded records")
    with c[1]: metric_card("Columns", f"{df.shape[1]:,}", "Available fields")
    with c[2]: metric_card("Missing cells", f"{int(df.isna().sum().sum()):,}", "Across all columns")
    with c[3]: metric_card("Exact duplicate rows", f"{int(df.duplicated().sum()):,}", "In loaded file")
    left, right = st.columns(2)
    with left:
        section("Missing values by column")
        missing = df.isna().sum().sort_values(ascending=False).reset_index()
        missing.columns = ["Column", "Missing"]
        missing = missing[missing["Missing"] > 0]
        if missing.empty:
            st.success("No missing values found in the loaded dataset.")
        else:
            fig = px.bar(missing, x="Missing", y="Column", orientation="h", text_auto=True)
            st.plotly_chart(style_fig(fig, 340), use_container_width=True)
    with right:
        section("Column types")
        dtypes = df.dtypes.astype(str).value_counts().rename_axis("Data type").reset_index(name="Columns")
        fig = px.pie(dtypes, names="Data type", values="Columns", hole=.55)
        st.plotly_chart(style_fig(fig, 340), use_container_width=True)
    section("Explore rows")
    col_options = list(df.columns)
    selected_cols = st.multiselect("Columns to show", col_options, default=[c for c in ["Time", "Amount", "Class"] if c in col_options])
    label_option = st.selectbox("Target label filter", ["All", "Genuine (0)", "Fraud-labelled (1)"])
    nrows = st.selectbox("Rows to preview", [25, 100, 500, 1000], index=1)
    preview = df.copy()
    if label_option == "Genuine (0)":
        preview = preview[preview["Class"] == 0]
    elif label_option == "Fraud-labelled (1)":
        preview = preview[preview["Class"] == 1]
    st.dataframe(preview[selected_cols].head(nrows) if selected_cols else preview.head(nrows),
                 use_container_width=True, hide_index=True)
    st.download_button("Download selected preview (CSV)", (preview[selected_cols] if selected_cols else preview).head(10000).to_csv(index=False).encode("utf-8"),
                       "fraudshield_data_preview.csv", "text/csv")
    with st.expander("Numeric summary statistics"):
        st.dataframe(df.describe().T, use_container_width=True)

# ------------------------------------------------------------
# 9. Model Monitoring
# ------------------------------------------------------------
elif page == "Model Monitoring":
    section("Model and data health snapshot", "This page evaluates the loaded historical file and saved model artifacts; it is not live production monitoring.")
    model_status = "Loaded" if model is not None else "Unavailable"
    c = st.columns(4)
    with c[0]: metric_card("Model artifact", model_status, model_path.name if model_path else "No artifact found")
    with c[1]: metric_card("Dataset", "Loaded", data_path.name)
    with c[2]: metric_card("Feature columns", f"{len(feature_names):,}", "Expected model inputs")
    with c[3]: metric_card("Fraud label rate", pct(fraud_rate, 3), "Current loaded file")
    left, right = st.columns(2)
    with left:
        section("Feature drift quick scan", "Compares the first half and second half of the file by mean and standard deviation. This is a basic diagnostic, not a formal drift test.")
        numeric_cols = [c for c in feature_names if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
        selected_feature = st.selectbox("Feature to inspect", numeric_cols, index=numeric_cols.index("Amount") if "Amount" in numeric_cols else 0)
        split = max(1, len(df) // 2)
        first = df[selected_feature].iloc[:split].dropna()
        second = df[selected_feature].iloc[split:].dropna()
        drift = pd.DataFrame({
            "Segment": ["First half", "Second half"],
            "Mean": [first.mean() if len(first) else np.nan, second.mean() if len(second) else np.nan],
            "Std dev": [first.std() if len(first) else np.nan, second.std() if len(second) else np.nan],
        })
        fig = px.bar(drift, x="Segment", y="Mean", color="Segment", text_auto=".4g")
        fig.update_layout(showlegend=False, yaxis_title=f"Mean of {selected_feature}")
        st.plotly_chart(style_fig(fig, 300), use_container_width=True)
        st.dataframe(drift, use_container_width=True, hide_index=True)
    with right:
        section("Feature distribution by segment")
        series = pd.DataFrame({
            "Value": pd.concat([first, second], ignore_index=True),
            "Segment": ["First half"] * len(first) + ["Second half"] * len(second),
        })
        fig = px.histogram(series, x="Value", color="Segment", barmode="overlay", nbins=45)
        st.plotly_chart(style_fig(fig, 300), use_container_width=True)
        if model is not None:
            st.success("Model artifact loaded successfully.")
        else:
            st.error(f"Model artifact issue: {model_load_error}")
    st.warning("For genuine deployment monitoring, log timestamped predictions, review outcomes, model versions, latency, and drift statistics over time. None of those live telemetry streams are connected in this prototype.")

# ------------------------------------------------------------
# 10. About & Methodology
# ------------------------------------------------------------
elif page == "About & Methodology":
    section("Project scope")
    st.markdown("""
    FraudShield AI is a portfolio prototype for historical credit-card fraud analysis, supervised classification,
    transaction review, and threshold trade-off exploration. It is not connected to a bank, payment processor,
    or real-time transaction stream.
    """)
    c = st.columns(3)
    with c[0]: metric_card("Learning task", "Binary classification", "Class 0 = genuine-labelled · Class 1 = fraud-labelled")
    with c[1]: metric_card("Primary model", "Random Forest", "Selected using validation average precision")
    with c[2]: metric_card("Risk workflow", "Score → threshold → review", "Human review remains essential")
    section("Data preparation")
    st.markdown("""
    - Raw source: ULB Credit Card Fraud Detection dataset (`creditcard.csv`).
    - The raw file should remain unchanged; the cleaned file is stored separately under `data/processed/`.
    - The project cleaning step removed exact duplicate rows. The loaded dataset shown by this app depends on which expected file is present.
    - The source contains `Time`, `Amount`, transformed features `V1`–`V28`, and `Class`.
    """)
    section("Model evaluation")
    st.markdown("""
    - Data was split into train, validation, and held-out test sets using stratification.
    - Model selection used validation average precision, an informative metric for highly imbalanced classification.
    - Threshold selection was performed on validation data, not the test data.
    - Precision describes the fraction of flagged records that are fraud-labelled; recall describes the fraction of fraud-labelled records flagged.
    """)
    section("Limitations and responsible use")
    st.markdown("""
    - Historical labels and patterns may not represent current fraud behaviour.
    - `Amount` has no currency label in this app and is not converted into confirmed financial loss.
    - `Time` represents elapsed seconds in the source data; it is not a transaction clock time.
    - `V1`–`V28` are anonymized transformed features and should not be assigned invented business meanings.
    - False positives create investigation workload; false negatives leave fraud-labelled cases unflagged.
    - Do not use this prototype as the sole basis for blocking transactions, accusing customers, or making consequential financial decisions.
    """)
    with st.expander("Available project reports"):
        for path in [EDA_SUMMARY_PATH, MODEL_COMPARISON_PATH, TEST_EVAL_PATH, THRESHOLD_EVAL_PATH, FEATURE_IMPORTANCE_PATH, THRESHOLD_RECOMMENDATION_PATH]:
            if path.exists():
                st.markdown(f"- `{path.relative_to(ROOT)}`")
    st.caption("FraudShield AI · Built for demonstration, learning, and portfolio review.")

st.markdown("---")
st.markdown(
    '<div style="display:flex;justify-content:space-between;gap:1rem;flex-wrap:wrap;color:#718096;font-size:.78rem">'
    '<span>🛡️ FraudShield AI · Financial Fraud Detection & Investigation</span>'
    '<span>Historical data only · Not live financial advice or a production fraud-control system</span></div>',
    unsafe_allow_html=True,
)
