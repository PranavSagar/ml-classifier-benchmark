from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"

TARGET = "y"
POSITIVE = "yes"

# duration is only known once the call has ended, so it leaks the outcome.
# The UCI notes say to drop it for any realistic predictive model.
LEAKY = ["duration"]

MODELS = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "kNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Random Forest (Ensemble)": "random_forest.joblib",
}


def read_bank_csv(path_or_buffer):
    # UCI ships this dataset semicolon-separated, but people re-export it with commas.
    df = pd.read_csv(path_or_buffer, sep=None, engine="python")
    df = df.drop(columns=[c for c in LEAKY if c in df.columns])

    # 999 in pdays is a sentinel for "never contacted before", not a real gap of
    # 999 days. Left as-is it dominates the scaler and the kNN distance metric.
    if "pdays" in df.columns:
        df["never_contacted"] = (df["pdays"] == 999).astype(int)
        df["pdays"] = df["pdays"].replace(999, 0)

    return df


def split_xy(df):
    y = (df[TARGET] == POSITIVE).astype(int)
    X = df.drop(columns=[TARGET])
    return X, y


def evaluate(y_true, y_pred, y_score):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_score),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
