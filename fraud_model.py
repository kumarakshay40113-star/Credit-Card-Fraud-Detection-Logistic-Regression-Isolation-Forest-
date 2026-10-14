"""Reusable training and prediction utilities for the fraud detection project."""
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    roc_auc_score, confusion_matrix
)

REQUIRED_TARGET = "Class"


def make_demo_dataset(n_samples=5000, fraud_rate=0.025, random_state=42):
    """Create synthetic data for app demonstration. Not real banking data."""
    rng = np.random.default_rng(random_state)
    n_fraud = max(2, int(n_samples * fraud_rate))
    n_normal = n_samples - n_fraud

    normal = rng.normal(0, 1, size=(n_normal, 28))
    fraud = rng.normal(0.25, 1.6, size=(n_fraud, 28))
    # Add a few shifted dimensions to make the toy data learnable.
    fraud[:, :5] += rng.choice([-2.5, 2.5], size=(n_fraud, 5))
    X = np.vstack([normal, fraud])
    y = np.concatenate([np.zeros(n_normal, dtype=int), np.ones(n_fraud, dtype=int)])

    amounts_normal = rng.lognormal(mean=3.1, sigma=1.0, size=n_normal)
    amounts_fraud = rng.lognormal(mean=3.5, sigma=1.2, size=n_fraud)
    amounts = np.concatenate([amounts_normal, amounts_fraud])
    times = rng.integers(0, 172800, size=n_samples)

    columns = {f"V{i+1}": X[:, i] for i in range(28)}
    df = pd.DataFrame(columns)
    df.insert(0, "Time", times)
    df["Amount"] = amounts
    df["Class"] = y

    # Shuffle rows reproducibly.
    return df.sample(frac=1, random_state=random_state).reset_index(drop=True)


def prepare_data(df):
    """Validate dataset and return numeric feature matrix and integer target."""
    if REQUIRED_TARGET not in df.columns:
        raise ValueError("CSV must contain a 'Class' column where 0=legitimate and 1=fraud.")
    data = df.copy()
    data = data.dropna(subset=[REQUIRED_TARGET])
    try:
        y = pd.to_numeric(data[REQUIRED_TARGET], errors="raise").astype(int)
    except Exception as exc:
        raise ValueError("The 'Class' column must contain numeric labels 0 and 1.") from exc

    if not set(y.unique()).issubset({0, 1}):
        raise ValueError("The 'Class' column must use only 0 (legitimate) and 1 (fraud).")

    X = data.drop(columns=[REQUIRED_TARGET])
    # Convert feature columns to numeric; remove non-numeric columns and missing/infinite rows.
    X = X.apply(pd.to_numeric, errors="coerce")
    valid = np.isfinite(X.to_numpy(dtype=float)).all(axis=1)
    X = X.loc[valid].reset_index(drop=True)
    y = y.loc[valid].reset_index(drop=True)

    if X.empty or X.shape[1] == 0:
        raise ValueError("No usable numeric feature columns were found.")
    if len(X) < 10:
        raise ValueError("Dataset is too small. Please provide at least 10 valid rows.")
    return X, y, list(X.columns)


def _metrics(y_true, y_pred, probabilities=None):
    out = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Fraud precision": precision_score(y_true, y_pred, zero_division=0),
        "Fraud recall": recall_score(y_true, y_pred, zero_division=0),
        "Fraud F1": f1_score(y_true, y_pred, zero_division=0),
    }
    if probabilities is not None and len(np.unique(y_true)) == 2:
        out["ROC-AUC"] = roc_auc_score(y_true, probabilities)
    return out


def train_and_evaluate(X, y, test_size=0.2, threshold=0.5,
                       contamination=0.01, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lr = LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=random_state
    )
    lr.fit(X_train_scaled, y_train)
    lr_prob = lr.predict_proba(X_test_scaled)[:, 1]
    lr_pred = (lr_prob >= threshold).astype(int)

    # Fit Isolation Forest on only legitimate training transactions.
    normal_train = X_train.loc[y_train.to_numpy() == 0]
    if len(normal_train) < 2:
        raise ValueError("Need at least two legitimate training transactions for Isolation Forest.")
    iso = IsolationForest(
        n_estimators=150,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1
    )
    iso.fit(scaler.transform(normal_train))
    iso_pred = (iso.predict(X_test_scaled) == -1).astype(int)

    combined_pred = np.logical_or(lr_pred == 1, iso_pred == 1).astype(int)

    metrics = {
        "Logistic Regression": _metrics(y_test, lr_pred, lr_prob),
        "Isolation Forest": _metrics(y_test, iso_pred),
        "Combined (OR)": _metrics(y_test, combined_pred),
    }

    test_results = X_test.copy().reset_index(drop=True)
    test_results["Actual Class"] = y_test.reset_index(drop=True)
    test_results["Fraud Probability"] = lr_prob
    test_results["Logistic Prediction"] = lr_pred
    test_results["Isolation Forest Prediction"] = iso_pred
    test_results["Combined Prediction"] = combined_pred
    test_results["Combined Status"] = np.where(
        combined_pred == 1, "Potential Fraud", "Not Flagged"
    )

    return {
        "scaler": scaler,
        "lr_model": lr,
        "if_model": iso,
        "X_test": X_test.reset_index(drop=True),
        "y_test": y_test.reset_index(drop=True),
        "lr_pred": lr_pred,
        "if_pred": iso_pred,
        "combined_pred": combined_pred,
        "lr_prob": lr_prob,
        "metrics": metrics,
        "test_results": test_results,
        "confusion_matrices": {
            "Logistic Regression": confusion_matrix(y_test, lr_pred, labels=[0, 1]),
            "Isolation Forest": confusion_matrix(y_test, iso_pred, labels=[0, 1]),
            "Combined (OR)": confusion_matrix(y_test, combined_pred, labels=[0, 1]),
        }
    }


def predict_transactions(X, scaler, lr_model, if_model, threshold=0.5):
    """Predict labels for a DataFrame containing the same feature columns as training."""
    X_scaled = scaler.transform(X)
    probability = lr_model.predict_proba(X_scaled)[:, 1]
    lr_pred = (probability >= threshold).astype(int)
    if_pred = (if_model.predict(X_scaled) == -1).astype(int)
    combined = np.logical_or(lr_pred == 1, if_pred == 1).astype(int)
    return pd.DataFrame({
        "Fraud Probability": probability,
        "Logistic Prediction": lr_pred,
        "Logistic Prediction Label": np.where(lr_pred == 1, "Potential fraud", "Legitimate"),
        "Isolation Forest Prediction": if_pred,
        "Isolation Forest Label": np.where(if_pred == 1, "Anomaly", "Normal"),
        "Combined Prediction": combined,
        "Combined Status": np.where(combined == 1, "Potential Fraud", "Not Flagged"),
    }, index=X.index)
