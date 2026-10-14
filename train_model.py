"""Train the two fraud detection models from creditcard.csv and save artifacts."""
from pathlib import Path
import joblib
import pandas as pd

from fraud_model import prepare_data, train_and_evaluate

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "creditcard.csv"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Download creditcard.csv from Kaggle "
            "and place it in the data folder. See README.md."
        )

    df = pd.read_csv(DATA_PATH)
    X, y, _ = prepare_data(df)
    result = train_and_evaluate(X, y)

    joblib.dump(result["scaler"], MODEL_DIR / "scaler.joblib")
    joblib.dump(result["lr_model"], MODEL_DIR / "logistic_regression.joblib")
    joblib.dump(result["if_model"], MODEL_DIR / "isolation_forest.joblib")
    result["test_results"].to_csv(OUTPUT_DIR / "fraud_predictions.csv", index=False)

    print("\nMODEL METRICS")
    for name, metrics in result["metrics"].items():
        print(f"\n{name}")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
    print(f"\nSaved models in: {MODEL_DIR}")
    print(f"Saved predictions in: {OUTPUT_DIR / 'fraud_predictions.csv'}")

if __name__ == "__main__":
    main()
