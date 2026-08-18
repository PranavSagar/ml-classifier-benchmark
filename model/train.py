import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import MODEL_DIR, MODELS, ROOT, evaluate, read_bank_csv, split_xy

SEED = 42
THRESHOLD = 0.5
RAW = ROOT / "data" / "bank-additional-full.csv"


def build_preprocessor(X):
    num = X.select_dtypes(include="number").columns.tolist()
    cat = X.select_dtypes(exclude="number").columns.tolist()
    return ColumnTransformer(
        [
            ("num", StandardScaler(), num),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
        ]
    )


def classifiers():
    return {
        # class_weight="balanced" wherever the estimator supports it: only 11% of
        # customers subscribe, so unweighted models just learn to say "no".
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=SEED
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8,
            min_samples_leaf=50,
            class_weight="balanced",
            random_state=SEED,
        ),
        "kNN": KNeighborsClassifier(n_neighbors=15, weights="distance", n_jobs=-1),
        "Naive Bayes": GaussianNB(),
        "Random Forest (Ensemble)": RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=20,
            class_weight="balanced_subsample",
            random_state=SEED,
            n_jobs=-1,
        ),
    }


def main():
    df = read_bank_csv(RAW)
    X, y = split_xy(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )

    # Ship the held-out split in its original raw form so that uploading it to the
    # app exercises the exact same preprocessing path as training did.
    MODEL_DIR.mkdir(exist_ok=True)
    raw = pd.read_csv(RAW, sep=";")
    raw.loc[X_test.index].to_csv(ROOT / "test_data.csv", sep=";", index=False)

    print(f"{len(df):,} rows | {X.shape[1]} features | positives {y.mean():.1%}")
    print(f"train {len(X_train):,} / test {len(X_test):,}\n")

    scores = {}
    for name, clf in classifiers().items():
        pipe = Pipeline([("prep", build_preprocessor(X_train)), ("clf", clf)])
        pipe.fit(X_train, y_train)

        # Score at an explicit 0.5 cutoff rather than pipe.predict(), so these numbers
        # line up exactly with the app's threshold slider at its default position.
        y_score = pipe.predict_proba(X_test)[:, 1]
        y_pred = (y_score >= THRESHOLD).astype(int)
        scores[name] = evaluate(y_test, y_pred, y_score)

        joblib.dump(pipe, MODEL_DIR / MODELS[name], compress=3)
        print(name, {k: round(v, 4) for k, v in scores[name].items()})

    (MODEL_DIR / "metrics.json").write_text(json.dumps(scores, indent=2))

    table = pd.DataFrame(scores).T.round(4)
    table.to_csv(MODEL_DIR / "metrics.csv")
    print("\n" + table.to_string())


if __name__ == "__main__":
    main()
