import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix, roc_curve

from common import MODEL_DIR, MODELS, ROOT, TARGET, evaluate, read_bank_csv, split_xy

st.set_page_config(page_title="Bank Marketing Classifier Benchmark", page_icon="📞", layout="wide")


@st.cache_resource
def load_models():
    return {name: joblib.load(MODEL_DIR / path) for name, path in MODELS.items()}


@st.cache_data
def score_all(df):
    X, y = split_xy(df)
    return {name: pipe.predict_proba(X)[:, 1] for name, pipe in load_models().items()}, y


def confusion_plot(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:,}", ha="center", va="center",
                color="white" if v > cm.max() / 2 else "black", fontsize=11)
    ax.set_xticks([0, 1], ["no", "yes"])
    ax.set_yticks([0, 1], ["no", "yes"])
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    fig.tight_layout()
    return fig


def roc_plot(y_true, scores, highlight):
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    for name, s in scores.items():
        fpr, tpr, _ = roc_curve(y_true, s)
        lead = name == highlight
        ax.plot(fpr, tpr, lw=2.2 if lead else 1, alpha=1 if lead else 0.35, label=name)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    return fig


st.title("Bank Marketing — Classifier Benchmark")
st.caption(
    "Five classifiers trained on the UCI Bank Marketing dataset, predicting whether a "
    "customer subscribes to a term deposit. Only 11% of customers say yes, so accuracy "
    "on its own is a trap — watch recall and MCC instead."
)

with st.sidebar:
    st.header("Data")
    upload = st.file_uploader("Upload test data (CSV)", type="csv")
    if upload:
        st.success(f"Using {upload.name}")
    else:
        st.info("Using the bundled `test_data.csv` (8,238 held-out rows).")
    st.download_button(
        "Download test_data.csv",
        (ROOT / "test_data.csv").read_bytes(),
        file_name="test_data.csv",
        mime="text/csv",
        help="Grab it, then upload it above to exercise the upload path.",
    )

    st.header("Model")
    choice = st.selectbox("Classifier", list(MODELS))

    st.header("Decision threshold")
    threshold = st.slider("Flag as 'yes' when probability ≥", 0.05, 0.95, 0.50, 0.05)
    st.caption("Lower it to chase more subscribers, raise it to waste fewer calls.")

df = read_bank_csv(upload if upload else ROOT / "test_data.csv")

if TARGET not in df.columns:
    st.error(f"No `{TARGET}` column found — the app needs the true labels to score the models.")
    st.stop()

scores, y = score_all(df)
rows = {name: evaluate(y, (s >= threshold).astype(int), s) for name, s in scores.items()}
table = pd.DataFrame(rows).T

leaderboard, detail, predictions = st.tabs(["Leaderboard", "Model detail", "Predictions"])

with leaderboard:
    st.subheader(f"All models at threshold {threshold:.2f}")
    st.dataframe(
        table.style.format("{:.4f}").highlight_max(axis=0, color="#d6f5d6"),
        width="stretch",
    )
    metric = st.radio("Compare on", list(table.columns), horizontal=True, index=5)
    st.bar_chart(table[metric], height=280)
    st.caption(
        f"Best {metric}: **{table[metric].idxmax()}** ({table[metric].max():.4f}). "
        "AUC is threshold-independent; every other column moves with the slider."
    )

with detail:
    st.subheader(choice)
    y_pred = (scores[choice] >= threshold).astype(int)

    cols = st.columns(6)
    for col, (name, value) in zip(cols, rows[choice].items()):
        col.metric(name, f"{value:.4f}")

    left, right = st.columns([1, 1.3])
    with left:
        st.markdown("**Confusion matrix**")
        st.pyplot(confusion_plot(y, y_pred))
    with right:
        st.markdown("**ROC curves**")
        st.pyplot(roc_plot(y, scores, choice))

    st.markdown("**Classification report**")
    report = classification_report(y, y_pred, target_names=["no", "yes"], output_dict=True)
    st.dataframe(pd.DataFrame(report).T.style.format("{:.4f}"), width="stretch")

with predictions:
    out = df.copy()
    out["probability"] = scores[choice].round(4)
    out["predicted"] = np.where(scores[choice] >= threshold, "yes", "no")
    flagged = (out["predicted"] == "yes").sum()

    st.subheader(f"{choice} flagged {flagged:,} of {len(out):,} customers to call")
    st.dataframe(out.head(200), width="stretch", height=420)
    st.download_button(
        "Download all predictions (CSV)",
        out.to_csv(index=False).encode(),
        file_name=f"predictions_{choice.split()[0].lower()}.csv",
        mime="text/csv",
    )
