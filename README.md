# Credit Card Fraud Detection (ML)

A beginner-friendly machine learning project combining **Logistic Regression** and **Isolation Forest**, with a Streamlit dashboard that runs locally in VS Code.

## Features

- Logistic Regression: supervised fraud classification using `Class` labels.
- Isolation Forest: anomaly detection, fitted only on legitimate training transactions.
- Combined rule: flags a transaction if either model flags it.
- Dashboard for dataset summary, class distribution, model metrics and confusion matrices.
- Transaction checker using held-out test records.
- Downloadable prediction CSV.
- Built-in **synthetic demo data** so the dashboard can be tested without downloading the Kaggle dataset.

> The built-in data is synthetic and is not real credit card data. This is an educational prototype, not a production payment system.

## Folder structure

```text
Credit_Card_Fraud_Detection_ML/
├── app.py
├── fraud_model.py
├── train_model.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── README.md
├── models/
│   └── .gitkeep
└── outputs/
    └── .gitkeep
```

## Requirements

- Python 3.10 or newer recommended
- Visual Studio Code
- Internet connection for the first-time package installation
- Kaggle dataset only if you want to train on the real benchmark data

## Run in VS Code

1. Extract the ZIP folder.
2. Open the extracted `Credit_Card_Fraud_Detection_ML` folder in VS Code.
3. Open **Terminal → New Terminal**.
4. Create and activate a virtual environment (recommended).

### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks activation, you can run the app without activating the environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The dashboard opens in your browser. If it does not, use the local URL printed in the terminal, usually `http://localhost:8501`.

## Use the built-in demo

- In the sidebar, select **Built-in demo data**.
- The app generates synthetic transactions automatically.
- The synthetic data is for demonstrating the workflow and interface only. Do not report its metrics as real-world performance.

## Use the Kaggle dataset

1. Download `creditcard.csv` from:
   https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. Put the file here:

```text
data/creditcard.csv
```

3. In the app sidebar select **Upload creditcard.csv** and upload it, or use the command-line training script below.

Expected target column: `Class` (`0` = legitimate, `1` = fraud). The common Kaggle dataset includes `Time`, `Amount`, and `V1` through `V28`.

### Train and save model files

After putting the dataset in `data/creditcard.csv`, run:

```powershell
python train_model.py
```

The script saves model artifacts to `models/` and predictions to `outputs/fraud_predictions.csv`.

## How the models work

### Logistic Regression
A supervised classifier trained on labeled examples. It outputs a fraud probability; the app uses a configurable threshold to decide whether to flag a transaction.

### Isolation Forest
An anomaly-detection model trained on legitimate training examples only. It flags feature patterns that appear unusual compared with those examples.

### Combined rule
The default combined prediction is positive if either Logistic Regression or Isolation Forest flags the transaction. This can improve recall but can also increase false positives. Always compare precision, recall, F1-score, ROC-AUC where applicable, and confusion matrices.

## Dataset and evaluation notes

- The dataset is strongly imbalanced: legitimate transactions greatly outnumber fraud cases.
- Accuracy alone can be misleading; pay particular attention to fraud precision and recall.
- The test split is held out from model fitting. Scaling is fit on the training data only.
- The Isolation Forest contamination setting is a model hyperparameter and should be tuned using validation data, not selected based on the final test set.
- The app's threshold and contamination controls let you observe how predictions and metrics change.

## Troubleshooting

- **`FileNotFoundError: creditcard.csv`**: for command-line training, make sure the CSV is in `data/creditcard.csv`. The app can run on built-in demo data without the file.
- **`ModuleNotFoundError`**: activate your virtual environment and run `pip install -r requirements.txt`.
- **`streamlit is not recognized`**: use `python -m streamlit run app.py`.
- **CSV upload error**: confirm it is a comma-separated CSV with numeric features and a `Class` column.

## Limitations and responsible use

This project is for coursework and demonstration. It is not validated for live financial decisions, does not connect to a bank or payment gateway, and should not be used to automatically block real customers. A production system would require privacy and security controls, robust validation, monitoring, drift detection, and human review.
