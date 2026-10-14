import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from fraud_model import (
    make_demo_dataset, prepare_data, train_and_evaluate,
    predict_transactions, REQUIRED_TARGET
)

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Credit Card Fraud Detection")
st.caption("Machine Learning project • Logistic Regression + Isolation Forest")
st.info(
    "Educational prototype only. A prediction is a risk flag, not proof of fraud. "
    "Do not use this demo to approve or decline real payments."
)

with st.sidebar:
    st.header("Dataset")
    data_mode = st.radio(
        "Choose data source",
        ["Built-in demo data", "Upload creditcard.csv"],
        help="Demo data is synthetic and is only for testing the app."
    )
    uploaded = None
    if data_mode == "Upload creditcard.csv":
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        st.caption("Expected columns: Time, V1–V28, Amount, Class.")
    st.divider()
    st.header("Model settings")
    test_size = st.slider("Test set (%)", 10, 40, 20, 5)
    threshold = st.slider(
        "Logistic Regression fraud threshold",
        min_value=0.10, max_value=0.90, value=0.50, step=0.05
    )
    contamination = st.slider(
        "Isolation Forest contamination",
        min_value=0.001, max_value=0.05, value=0.01, step=0.001,
        format="%.3f"
    )
    run_button = st.button("Train / refresh models", type="primary", use_container_width=True)

@st.cache_data(show_spinner=False)
def demo_data():
    return make_demo_dataset(n_samples=5000, fraud_rate=0.025, random_state=42)

if data_mode == "Upload creditcard.csv":
    if uploaded is None:
        st.warning("Upload your Kaggle creditcard.csv file in the sidebar, or switch to Built-in demo data.")
        st.stop()
    try:
        df = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"Could not read the CSV: {exc}")
        st.stop()
    dataset_label = "Uploaded dataset"
else:
    df = demo_data()
    dataset_label = "Synthetic demo dataset"

if REQUIRED_TARGET not in df.columns:
    st.error("The dataset must contain a 'Class' column (0 = legitimate, 1 = fraud).")
    st.stop()

try:
    X, y, feature_names = prepare_data(df)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

if y.nunique() < 2 or y.value_counts().min() < 2:
    st.error("The dataset needs at least two examples of each class to create a stratified train/test split.")
    st.stop()

if run_button or "model_result" not in st.session_state or st.session_state.get("dataset_label") != dataset_label:
    with st.spinner("Training models and evaluating predictions..."):
        st.session_state["model_result"] = train_and_evaluate(
            X, y, test_size=test_size / 100,
            threshold=threshold, contamination=contamination,
            random_state=42
        )
        st.session_state["dataset_label"] = dataset_label
        st.session_state["data_fingerprint"] = (len(df), tuple(df.columns))

result = st.session_state["model_result"]
# Retrain when the settings are changed, even if the button was not clicked.
settings = (test_size, threshold, contamination, dataset_label, len(df), tuple(df.columns))
if st.session_state.get("settings") != settings:
    with st.spinner("Updating model settings..."):
        result = train_and_evaluate(
            X, y, test_size=test_size / 100,
            threshold=threshold, contamination=contamination,
            random_state=42
        )
        st.session_state["model_result"] = result
        st.session_state["settings"] = settings

test_y = result["y_test"]
lr_pred = result["lr_pred"]
if_pred = result["if_pred"]
combined_pred = result["combined_pred"]
n_total = len(df)
fraud_count = int((y == 1).sum())
fraud_rate = fraud_count / len(y) * 100

m1, m2, m3, m4 = st.columns(4)
m1.metric("Transactions", f"{n_total:,}")
m2.metric("Known fraud labels", f"{fraud_count:,}")
m3.metric("Fraud share", f"{fraud_rate:.2f}%")
m4.metric("Test records", f"{len(test_y):,}")

st.caption(f"Data source: **{dataset_label}**. {('Synthetic values — not real card transactions.' if data_mode == 'Built-in demo data' else 'Treat uploaded data as sensitive.')}")
st.divider()

tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Dashboard", "🧠 Model comparison", "🔎 Transaction checker", "📁 Data preview"]
)

with tab1:
    st.subheader("Dataset overview")
    left, right = st.columns(2)
    with left:
        counts = y.value_counts().reindex([0, 1], fill_value=0)
        fig, ax = plt.subplots()
        ax.bar(["Legitimate (0)", "Fraud (1)"], counts.values)
        ax.set_ylabel("Number of transactions")
        ax.set_title("Class distribution")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    with right:
        amount_col = "Amount" if "Amount" in df.columns else None
        if amount_col:
            fig, ax = plt.subplots()
            normal_amounts = df.loc[y == 0, amount_col].clip(lower=0)
            fraud_amounts = df.loc[y == 1, amount_col].clip(lower=0)
            ax.hist(normal_amounts, bins=30, alpha=0.7, label="Legitimate")
            ax.hist(fraud_amounts, bins=30, alpha=0.7, label="Fraud")
            ax.set_xlabel("Transaction amount")
            ax.set_ylabel("Count")
            ax.set_title("Amount distribution (clipped to display)")
            ax.legend()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        else:
            st.write("No `Amount` column found in this dataset.")
    st.subheader("Combined model evaluation")
    st.write("The combined model flags a transaction if **either** algorithm flags it.")
    metrics = result["metrics"]
    metric_df = pd.DataFrame(metrics).T
    st.dataframe(metric_df.style.format("{:.3f}"), use_container_width=True)
    st.caption("Precision, recall and F1 are calculated for the fraud class. ROC-AUC is reported for Logistic Regression only.")

with tab2:
    st.subheader("Confusion matrices")
    col1, col2, col3 = st.columns(3)
    for col, name, pred in [
        (col1, "Logistic Regression", lr_pred),
        (col2, "Isolation Forest", if_pred),
        (col3, "Combined", combined_pred)
    ]:
        with col:
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(test_y, pred, labels=[0, 1])
            st.markdown(f"**{name}**")
            st.dataframe(
                pd.DataFrame(cm, index=["Actual normal", "Actual fraud"],
                             columns=["Predicted normal", "Predicted fraud"]),
                use_container_width=True
            )
    st.markdown("**How to interpret the metrics**")
    st.markdown(
        "- **Precision:** Of the transactions flagged as fraud, how many were actually fraud.\n"
        "- **Recall:** Of all actual fraud transactions, how many the model detected.\n"
        "- **F1-score:** Balance between precision and recall.\n"
        "- **False positives:** Legitimate transactions incorrectly flagged. These can inconvenience customers."
    )
    st.warning("The OR-combination often increases detections but may also increase false positives. Compare the metrics before drawing conclusions.")

with tab3:
    st.subheader("Check a transaction from the test set")
    st.write("Choose one held-out test transaction. This shows how the trained models classify its feature values.")
    test_positions = list(range(len(result["X_test"])))
    selected = st.number_input(
        "Test row index", min_value=0, max_value=max(0, len(test_positions)-1),
        value=0, step=1
    )
    ix = int(selected)
    one_x = result["X_test"].iloc[[ix]]
    one_result = predict_transactions(
        one_x, result["scaler"], result["lr_model"], result["if_model"],
        threshold=threshold
    ).iloc[0]
    actual = int(result["y_test"].iloc[ix])
    c1, c2, c3 = st.columns(3)
    c1.metric("Actual label", "Fraud" if actual == 1 else "Legitimate")
    c2.metric("Logistic Regression", "Potential fraud" if one_result["Logistic Prediction"] else "Legitimate")
    c3.metric("Isolation Forest", "Anomaly" if one_result["Isolation Forest Prediction"] else "Normal")
    if one_result["Combined Prediction"]:
        st.error("Combined result: POTENTIAL FRAUD — review this transaction.")
    else:
        st.success("Combined result: NOT FLAGGED by these models.")
    st.metric("Logistic Regression fraud probability", f"{one_result['Fraud Probability']:.1%}")
    with st.expander("View transaction features"):
        st.dataframe(one_x, use_container_width=True)

    export = result["test_results"].to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download test predictions CSV",
        data=export,
        file_name="fraud_predictions.csv",
        mime="text/csv"
    )

with tab4:
    st.subheader("Dataset preview")
    st.dataframe(df.head(100), use_container_width=True)
    st.write("Column names:", ", ".join(map(str, df.columns)))
    st.download_button(
        "Download loaded dataset",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="loaded_transactions.csv",
        mime="text/csv"
    )

st.divider()
st.caption("Student ML project • Logistic Regression + Isolation Forest • For learning and demonstration only")
