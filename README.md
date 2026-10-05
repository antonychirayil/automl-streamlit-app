# AutoML App

A Streamlit web app that takes a raw CSV and walks it all the way to a
trained, tuned, explained model — without writing any code for each new
dataset.

Upload a file, pick the column you want to predict, and the app works out
whether that's a classification or a regression problem, preprocesses the
data, trains and compares five models, searches for better
hyperparameters, charts the results, and will then score a second, unseen
file with whichever model won.

---

## Quick start

```bash
# 1. create an environment (Python 3.10 – 3.13)
py -3.13 -m venv venv313
venv313\Scripts\activate          # Windows
# source venv313/bin/activate     # macOS / Linux

# 2. install
pip install -r requirements.txt

# 3. run
streamlit run app.py
```

The app opens at <http://localhost:8501>.

Optional one-line EDA reports (ydata-profiling and Sweetviz) are a
separate install — see [Optional extras](#optional-extras) below.

---

## What it does

| Step | What happens |
| --- | --- |
| **1. Dataset** | Upload a CSV. The app reports rows, columns, missing values and duplicates. |
| **2. Target variable** | Pick the column to predict. |
| **3. Preprocess** | Detects classification vs. regression, splits train/test, fills missing values, encodes categories, scales numbers, and keeps the *k* most predictive features. |
| **4. Train models** | Fits five models, scores them on the test set, and cross-validates each one on the training set. |
| **5. Tune hyperparameters** | Grid Search or Random Search per model, then re-scores the winners. |
| **6. Predict on new data** | Scores a second CSV through the same fitted pipeline, with confidence scores and a drift check. |
| **7. Automated EDA** | Optional full profiling report from ydata-profiling or Sweetviz. |

Results download as CSV, charts as PNG, and the winning model as a
`.pkl` you can load elsewhere with `joblib.load()`.

### Models compared

Five per problem type, chosen to span genuinely different approaches — a
linear model, a single tree, an ensemble of trees, a margin-based model
and a distance-based one — so the comparison says something about the
data rather than comparing five variations of one idea.

| Classification | Regression |
| --- | --- |
| Logistic Regression | Linear Regression |
| Decision Tree | Decision Tree |
| Random Forest | Random Forest |
| SVM | SVR |
| KNN | KNN |

**Metrics.** Classification reports accuracy, precision, recall, F1
(weighted) and ROC-AUC; regression reports MAE, MSE, RMSE and R². Both
also report a cross-validated mean and standard deviation, so you can see
how much each score would move on a different split — a single number
from a single split can be lucky. Winners are chosen on F1 for
classification (it balances precision against recall, which plain
accuracy won't do on an imbalanced target) and on R² for regression.

---

## Project structure

```
automl_app/
├── app.py                  entry point — sets up the page, calls the UI
├── config.py               every tunable number, with the reasoning
├── state.py                session state + cached expensive work
├── ui_sidebar.py           all input controls   (WRITES session state)
├── ui_main.py              all output rendering (READS session state)
├── preprocessing.py        splitting, imputing, encoding, scaling, selection
├── models.py               model definitions, training, hyperparameter search
├── evaluation.py           metrics, comparison table, picking a winner
├── visualization.py        matplotlib figures
├── profiling.py            optional EDA reports (ydata-profiling / sweetviz)
├── tests/                  pytest suite
├── requirements.txt        core dependencies
└── requirements-extras.txt optional EDA dependencies
```

Two conventions hold the layout together.

**The bottom five modules import no Streamlit at all.** They're plain
Python that takes data and returns results, which is what makes them
testable without a browser and reusable in a notebook or a script. All
the Streamlit-specific machinery lives in `app.py`, `state.py` and the
two `ui_*` modules.

**Input and output are separated.** `ui_sidebar.py` writes to session
state; `ui_main.py` reads from it and renders. So "where does this value
come from?" is answered in one file and "where is this shown?" in the
other, and a value can only ever be set in one place.

---

## How the pipeline works

**Everything is fitted on the training data only.** The imputer learns
its median, the scaler learns its mean and standard deviation, the
encoder learns its categories and the selector learns which features
matter — all from the training set, and all merely *applied* to the test
set afterwards. Fitting on data you later evaluate against leaks
information from it into the model, and leakage doesn't announce itself:
it just quietly inflates every score you report.

That same fitted pipeline is what `transform_new_data()` reuses when you
upload a second file, so predictions on new data go through exactly the
transformations the model was trained under.

**Classification splits are stratified**, keeping the class balance the
same on both sides — without it, a random split of a 90/10 target can
dump most of the rare class into one side by chance. When a class is too
rare to stratify (fewer than two examples) the app falls back to a plain
split and says so, rather than crashing.

**Cross-validation folds shrink to fit the data.** Cross-validating needs
every class to appear at least once per fold; with an imbalanced target
the rarest class can fall short. Rather than let the search crash, the
number of folds is reduced to what the data supports, or tuning is
skipped for that model with a note explaining why.

**SVMs are trained on a capped subsample.** They scale roughly
quadratically with row count, so on tens of thousands of rows a single
fit — let alone a cross-validated search — takes a very long time. SVM
and SVR are fitted on at most `SVM_MAX_TRAIN_ROWS` rows, but still
*evaluated* on the full test set. The app reports when this happened.

---

## Tests

```bash
pip install pytest
pytest tests/ -q
```

60 tests covering problem-type detection, splitting, the preprocessing
pipeline, transforming new data, metrics, model wiring, cross-validation
safety and the optional-extras helpers.

They run against real scikit-learn on small synthetic datasets, so the
whole suite finishes in seconds. Two are worth knowing about:
`test_scaling_is_fitted_on_training_data_only` is the anti-leakage guard,
and `test_a_learnable_dataset_actually_gets_learned` builds data with a
known signal and asserts the pipeline finds it — if that fails, something
upstream is broken rather than the model being bad.

The tests for `profiling.py` deliberately pass whether or not the
optional libraries are installed, since behaving sensibly in both states
is the point of that module.

---

## Optional extras

Automated EDA reports come from two libraries that are kept out of
`requirements.txt` because they're heavy and one is fussy about Python
versions. The app runs perfectly without them — Step 7 just shows them as
"Not installed".

```bash
pip install -r requirements-extras.txt
```

- **Sweetviz** installs anywhere, and colours every chart by how the
  feature relates to your target (numeric or boolean targets only).
- **ydata-profiling** — the library formerly called pandas-profiling — is
  the more exhaustive of the two, and requires **Python 3.10–3.13**. It
  will not install on 3.14.

---

## Troubleshooting

**`ImportError` after editing a module.** Streamlit re-runs `app.py` on
every interaction, but it does *not* re-import modules that are already
loaded. After changing anything other than `app.py`, stop the server with
`Ctrl+C` and run `streamlit run app.py` again.

**`No module named 'pkg_resources'` when generating a ydata-profiling
report.** Two changes collided: Python 3.12 stopped seeding new virtual
environments with setuptools, and setuptools 82.0 then removed
`pkg_resources` altogether — while ydata-profiling still imports it. A
plain `pip install setuptools` therefore doesn't fix it. Pin below 82:

```bash
pip install "setuptools<82"
```

**ydata-profiling won't install.** Check your Python version with
`python --version`. It caps at 3.13; on 3.14 pip refuses it, and
`requirements-extras.txt` skips that line automatically.

**Hyperparameter tuning is slow.** Lower `CV_FOLDS` or trim the grids in
`models.py`. The Random Forest grid is already capped deliberately —
unbounded tree depth was taking about 120 seconds of a 135-second run for
a negligible score difference.

---

## Configuration

Every threshold, cap and colour lives in `config.py` with a comment
explaining the reasoning — sample sizes, cross-validation folds, the
confidence threshold below which a prediction gets flagged, the drift
warning trigger, chart colours. Change behaviour there rather than
hunting through the modules.
