# Bank Marketing — Classifier Benchmark

Five classification models trained on the UCI Bank Marketing dataset, compared on six
metrics, and served through an interactive Streamlit app.

- **GitHub repository:** https://github.com/PranavSagar/ml-classifier-benchmark
- **Live app:** _(Streamlit Community Cloud link — see Deployment below)_

---

## 1. Problem statement

A Portuguese retail bank ran direct-marketing telephone campaigns to sell term deposits.
Each row is one call to one customer, and the target is whether that customer subscribed.

Calling everyone is expensive, so the business question is: **given what the bank knows
about a customer before it dials, can we predict who will subscribe?** A useful model has
to find the subscribers (recall) without flagging so many non-subscribers that the call
list becomes worthless (precision).

The catch is that only **11.3%** of customers subscribe. A model that predicts "no" for
everybody scores **88.7% accuracy** and is completely useless. That single fact drives
every choice below — which metrics we rank on, and why the models are class-weighted.

Two decisions made before any model was trained:

- **`duration` is dropped.** It records how long the call lasted, which is only known
  *after* the call is over — and a call that ends in a sale runs long. Keeping it pushes
  accuracy above 90% by leaking the answer. The UCI documentation says to discard it for
  any realistic predictive model, so it is dropped in `common.py`.
- **`pdays = 999` is a sentinel**, not a real value. It means "never contacted before".
  Left as-is it dominates the scaler and completely distorts the kNN distance metric, so
  it is split into a `never_contacted` flag with the numeric value set to 0.

## 2. Dataset description

| | |
|---|---|
| **Source** | [UCI ML Repository — Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing) (`bank-additional-full.csv`) |
| **Instances** | 41,188 |
| **Features used** | 20 (19 original after dropping `duration`, plus the derived `never_contacted`) |
| **Target** | `y` — did the customer subscribe to a term deposit (`yes` / `no`) |
| **Class balance** | 11.3% `yes` / 88.7% `no` — heavily imbalanced |
| **Missing values** | None, but 12,718 cells carry the literal string `unknown`, kept as its own category |
| **Train / test split** | 32,950 / 8,238, stratified, `random_state=42` |

**Numeric (10):** `age`, `campaign`, `pdays`, `previous`, `emp.var.rate`, `cons.price.idx`,
`cons.conf.idx`, `euribor3m`, `nr.employed`, `never_contacted`

**Categorical (10):** `job`, `marital`, `education`, `default`, `housing`, `loan`,
`contact`, `month`, `day_of_week`, `poutcome`

The last five numeric columns are macro-economic indicators for the month of the call
(employment variation, consumer price and confidence indices, the 3-month Euribor rate,
and the number employed). They matter a lot — and they move together, which turns out to
be Naive Bayes' undoing.

`test_data.csv` in this repo is the held-out 8,238-row split in its original raw form.
Upload it to the app, or leave the uploader empty and the app loads it by default.

## 3. Preprocessing

Every model is a single scikit-learn `Pipeline`, so preprocessing is fitted on the
training fold only and travels with the saved model — the app just calls `predict_proba`
and cannot leak or drift.

```
ColumnTransformer
├── numeric      → StandardScaler
└── categorical  → OneHotEncoder(handle_unknown="ignore")
        ↓
    classifier
```

`handle_unknown="ignore"` matters for deployment: if an uploaded CSV contains a job title
the model never saw, the app degrades to zeros for that column instead of crashing.

**`class_weight="balanced"` is set wherever the estimator supports it** (Logistic
Regression, Decision Tree, Random Forest). Without it, all three collapse to predicting
"no" almost everywhere — high accuracy, near-zero recall, useless. kNN and GaussianNB have
no equivalent knob, and the results show exactly what that costs them.

## 4. Models used

| Model | Key settings | Why |
|---|---|---|
| Logistic Regression | `max_iter=2000`, `class_weight="balanced"` | Linear baseline; interpretable coefficients |
| Decision Tree | `max_depth=8`, `min_samples_leaf=50`, `class_weight="balanced"` | Unconstrained it memorised the training set |
| kNN | `n_neighbors=15`, `weights="distance"` | k=5 gave AUC 0.7334 vs 0.7570 at k=15; k=25 edges AUC to 0.7611 but recall keeps sliding (0.318 → 0.274 → 0.260), so 15 is the compromise |
| Naive Bayes | `GaussianNB` | Fastest baseline; tests how badly the independence assumption hurts here |
| Random Forest | `n_estimators=200`, `min_samples_leaf=20`, `class_weight="balanced_subsample"` | Bagged trees to cut the single tree's variance |

### Comparison table

All models evaluated on the same 8,238-row held-out test set, at a 0.50 decision
threshold. Best value in each column is **bold**.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8354 | 0.8010 | 0.3684 | 0.6455 | 0.4691 | 0.4011 |
| Decision Tree | 0.8224 | 0.7993 | 0.3476 | **0.6573** | 0.4547 | 0.3866 |
| kNN | **0.8954** | 0.7570 | **0.5747** | 0.2737 | 0.3708 | 0.3480 |
| Naive Bayes | 0.8224 | 0.7756 | 0.3372 | 0.5970 | 0.4310 | 0.3545 |
| **Random Forest (Ensemble)** | 0.8587 | **0.8109** | 0.4171 | 0.6401 | **0.5051** | **0.4403** |

Reproduce with `python model/train.py`; the numbers are written to `model/metrics.json`
and `model/metrics.csv`.

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| **Logistic Regression** | Strong showing for the simplest model — second-best MCC (0.4011) and AUC within 0.01 of the winner. The signal here is largely monotonic (as `euribor3m` and `nr.employed` fall, subscriptions rise), which is exactly what a linear boundary captures well. Cheapest to serve, and the only model whose coefficients you could hand to a compliance reviewer. |
| **Decision Tree** | The widest net: best recall of all five (0.6573) but the second-worst precision (0.3476), so roughly two of every three customers it flags will decline. Depth had to be capped at 8: unconstrained, it hit **0.9946 training accuracy and collapsed to 0.6194 test AUC** — memorising the training set cost it 18 points of AUC against the capped version's 0.7993. Matches logistic regression on AUC but with visibly more variance, which is precisely what the ensemble fixes. |
| **kNN** | The cautionary tale of this dataset. It posts the **highest accuracy of all five models (0.8954) and is the worst model in the set.** Predicting "no" for everyone already scores 0.8874, so kNN's headline number beats the do-nothing baseline by 0.8 points. Recall of 0.2737 means it misses nearly three of every four subscribers, and its AUC (0.7570) is the lowest here. It has no `class_weight` option, so the majority class simply outvotes the minority in every neighbourhood — and it is the costliest model to serve, since it carries the whole training set to inference. |
| **Naive Bayes** | Its independence assumption is badly violated. `emp.var.rate`, `euribor3m` and `nr.employed` are near-collinear macro-economic indicators, so NB multiplies essentially the same piece of evidence three times and becomes overconfident. Combined with one-hot columns, which are mutually exclusive rather than independent, it lands the second-lowest MCC (0.3545) despite reasonable recall. Fastest to train by a wide margin. |
| **Random Forest (Ensemble)** | Best on every metric that accounts for both classes: AUC 0.8109, F1 0.5051, MCC 0.4403. It keeps almost all of the single tree's recall (0.6401 vs 0.6573) while lifting precision from 0.3476 to 0.4171 — bagging removes the single tree's variance without giving up its sensitivity. Note its accuracy (0.8587) is *lower* than kNN's, which is the whole point. |
| **Overall winner for your dataset?** | **Random Forest.** On an 11%-positive target, accuracy is the wrong yardstick — it crowns kNN, which misses 73% of the subscribers the campaign exists to find. Ranking on MCC and AUC, which are both sensitive to performance on the minority class, puts Random Forest first, and it wins F1 as well. Logistic regression is the honourable mention: 0.039 behind on MCC for a fraction of the inference cost, and worth choosing if interpretability outranks the last few points. |

**A note on the threshold.** Every metric except AUC in the table above depends on the
0.50 cutoff, which is an arbitrary default rather than a business decision. The app's
threshold slider exposes this directly: drop Random Forest to 0.30 and recall climbs from
0.6401 to 0.8028 while precision falls from 0.4171 to 0.2029 — the right trade if a missed
subscriber costs more than a wasted call, and the wrong one if it does not. AUC is the only column in the table that is threshold-independent
— which is why it, along with MCC, carries the most weight in the verdict above.

## 5. The Streamlit app

Three tabs over whichever CSV is loaded:

- **Leaderboard** — all five models scored side by side at the current threshold, best
  value in each column highlighted, plus a bar chart on any metric you pick.
- **Model detail** — the six metrics for the selected model, its confusion matrix, a
  full classification report, and ROC curves for all five with the selection highlighted.
- **Predictions** — per-customer subscription probability and predicted label, with the
  full scored file downloadable as CSV.

Sidebar controls: **CSV upload**, **model dropdown**, and the **decision threshold
slider**. With the uploader empty it falls back to the bundled `test_data.csv`. The
uploader sniffs the delimiter, so both the semicolon-separated UCI format and a
comma-separated re-export work.

## 6. Project structure

```
ml-classifier-benchmark/
├── app.py                  Streamlit app
├── common.py               shared loading / cleaning / metrics
├── requirements.txt
├── test_data.csv           held-out test split, raw form
├── data/
│   └── bank-additional-full.csv
└── model/
    ├── train.py            trains all five, writes the artefacts below
    ├── *.joblib            five fitted pipelines
    ├── metrics.json
    └── metrics.csv
```

## 7. Running it locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python model/train.py     # optional — fitted pipelines are already committed
streamlit run app.py
```

`scikit-learn` and `numpy` are both pinned in `requirements.txt` to the versions the
committed pipelines were fitted with (1.9.0 and 2.4.6). Unpinning scikit-learn makes
joblib emit version warnings and risks the app failing to load the models after a release
— and the version genuinely moves the numbers, since the Random Forest's MCC shifted from
0.4426 to 0.4403 between 1.6.1 and 1.9.0 purely from a changed RNG stream. Pinning numpy
is subtler but matters just as much: fitting under numpy 2.0 instead of 2.4 changed kNN's
AUC in the fourth decimal, because a different BLAS resolves distance ties differently.
Without both pins the deployed app would quietly disagree with the comparison table above.

Both pins are chosen to have prebuilt wheels for **Python 3.11 through 3.14**, so the app
builds on whichever interpreter Streamlit Community Cloud provisions. `scikit-learn 1.6.1`
stops at cp313: on Streamlit's current Python 3.14 image it falls back to compiling from
source and the build times out.

## 8. Deployment

Deployed on Streamlit Community Cloud from this repository — `main` branch, `app.py`.
The five fitted pipelines total ~8 MB and are committed to the repo, so the
container loads them at startup and does no training. `random_forest.joblib` is
`min_samples_leaf=20` and `compress=3`; at the default `min_samples_leaf=5` it serialised
to 69 MB, which the free tier's memory limit does not tolerate.

---

*BITS Pilani WILP — M.Tech (AIML), Machine Learning, Assignment 2.*
