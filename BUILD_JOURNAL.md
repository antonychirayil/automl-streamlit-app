# Build Journal — AutoML Streamlit App

How this project was built, stage by stage, from an empty folder to the
current structure. Written as a learning record: each stage lists what
was added, the Streamlit or ML concept behind it, and — where it matters
— what went wrong and what that taught.

Dataset used throughout: `train_LZdllcl.csv`, an HR analytics file of
**54,808 rows × 14 columns**, predicting `is_promoted`.

---

## Stage 0 — Environment, and two false starts

Before any code, two decisions had to be unwound.

**The cloud sandbox couldn't install packages.** The first plan was to
build in a cloud environment, but PyPI was blocked there. The project
moved onto the local Windows machine, with files synced across —
everything from here on runs at
`C:\Users\MYPC\Documents\Projects\automl_app`.

**Python 3.14.6 was too new.** `pandas==2.2.2` had no prebuilt wheel for
3.14, so pip tried to compile it from source and failed for want of
Visual Studio. Fix: loosen `requirements.txt` from exact pins (`==`) to
minimums (`>=`), letting pip choose versions that actually have wheels.

> **Lesson.** A brand-new Python release is a liability for data work.
> The ecosystem lags the language by 6–12 months. This cost time twice —
> it comes back in Stage 12.

**Files:** `requirements.txt`, `.gitignore`, `venv/`

---

## Stage 1 — Hello, Streamlit

A single `app.py`, one `st.title()`, and `streamlit run app.py`.

**Concept: the rerun model.** Streamlit re-executes your entire script
from the top every time anyone touches any widget. There is no callback
system and no event loop — the script *is* the UI, and it runs again and
again. Almost every design decision later follows from this one fact.

---

## Stage 2 — Upload a CSV, pick a target

`st.file_uploader()` for the file, `st.selectbox()` for the column to
predict, `st.dataframe()` to preview it.

**Concept: `st.session_state`.** Because the script re-runs constantly,
an ordinary Python variable dies on every interaction. `st.session_state`
is the dictionary that survives reruns — so the uploaded dataframe goes
in there, not in a local variable.

---

## Stage 3 — Detect the problem type

**New file: `preprocessing.py`** — the first split of logic away from UI.

`detect_problem_type()` decides classification vs. regression with
explainable rules rather than a guess: a non-numeric column is always
classification (you can't average "north" and "south"); a numeric column
is classification when it has few distinct values, because a column of
0/1 flags or 1–5 ratings is a set of labels that happen to be stored as
numbers.

**Concept: `st.button()` fires once.** A button returns `True` only on
the single rerun caused by its own click — on the very next rerun it's
`False` again. So anything a button produces must be written into session
state immediately, or it vanishes.

---

## Stage 4 — The preprocessing pipeline

`split_data()` and `preprocess_and_select()`, built on scikit-learn's
`ColumnTransformer` and `Pipeline`:

- numeric columns → `SimpleImputer(median)` → `StandardScaler`
- categorical columns → `SimpleImputer(most_frequent)` → `OneHotEncoder`
- then `SelectKBest` keeps the most predictive features

On this dataset: **43,846 training rows / 10,962 test rows**, 59
engineered columns after encoding.

**Concept: fit on training data only.** Every transformer learns its
statistics — the median, the mean and standard deviation, the category
list — from the training set alone, and is then merely *applied* to the
test set. Fitting on data you later evaluate against leaks information
into the model, and leakage doesn't announce itself: it silently inflates
every score you report.

**Concept: stratified splitting.** Classification splits preserve the
class balance on both sides. Without it, a random split of a 90/10 target
can dump most of the rare class into one side by luck.

---

## Stage 5 — Train and compare five models

**New files: `models.py`, `evaluation.py`**

Five models per problem type, chosen to span genuinely different
approaches rather than five variants of one idea:

| Classification | Regression |
| --- | --- |
| Logistic Regression | Linear Regression |
| Decision Tree | Decision Tree |
| Random Forest | Random Forest |
| SVM | SVR |
| KNN | KNN |

`evaluation.py` computes accuracy/precision/recall/F1 for classification
and MAE/MSE/RMSE/R² for regression, builds a comparison table, and names
a winner — on **F1**, not accuracy, because accuracy flatters a model
that just predicts the majority class on an imbalanced target.

**Result:** Random Forest won with **F1 = 0.9044**.

**Performance decision:** SVMs scale roughly quadratically with row
count, so on 44k rows a single fit crawls. SVM/SVR are now fitted on a
capped random subsample (5,000 rows) but still *evaluated* on the full
test set — and the app says when it did this.

---

## Stage 6 — Hyperparameter tuning

`get_param_grids()` and `tune_and_evaluate()`: exhaustive `GridSearchCV`
while a grid is small, `RandomizedSearchCV` once it grows past 20
combinations.

**Result:** tuned Random Forest scored **F1 = 0.9036** — *0.0008 below*
the baseline.

> **Lesson.** This is not a bug. Tuning optimises for average performance
> across cross-validation folds, not for one particular test split. When
> the signal in the data is weak, the difference lands inside the noise.
> A tuned model scoring fractionally below its baseline is a normal,
> honest result.

**Performance fix:** the first Random Forest grid took **117 of 135
total tuning seconds**. Capping `n_estimators` at 150 and dropping
unlimited tree depth cut it to about **20 seconds** with negligible score
change. The reasoning is recorded in a comment so the cap doesn't look
arbitrary later.

---

## Stage 7 — Visualizations

**New file: `visualization.py`** — five charts across `st.tabs()`:
feature distributions, correlation heatmap, model comparison, feature
importance, and confusion matrix (or predicted-vs-actual for regression).

**Concept: build figures, never `plt.show()`.** There is no screen to
open a window on — the app is a web page. Every function returns a
matplotlib `Figure`, and the caller passes it to `st.pyplot()`. That also
makes the charts reusable in a notebook, and testable.

Functions return `None` when a chart doesn't apply (no numeric columns;
a KNN model, which has no notion of feature importance) so the UI can
explain why instead of showing an empty frame.

---

## Stage 8 — Robustness pass

The first "make it professional" pass, aimed at everything that would
crash the app with a raw traceback.

- **New exception `DataValidationError`** — separates *"your data has a
  problem you can fix"* from *"the code has a bug"*. The app catches it
  specifically and prints the message as a plain sentence.
- Guards for an empty target, a constant target, and columns that are
  100% missing.
- **Rare-class safety:** stratification falls back to a plain split (with
  a note) when a class has fewer than 2 members; `_effective_cv()`
  shrinks the cross-validation fold count to what the rarest class
  supports, or skips tuning with an explanation.
- **Caching bug fixed:** the 3.6MB CSV was being re-parsed on *every*
  rerun. `@st.cache_data` on the loader, plus a `(name, size)` file
  signature to detect a genuinely new upload.

**Concept: `@st.cache_data`.** Memoises against the function's arguments.
Given that every click re-runs the whole script, this is the difference
between an app that feels instant and one that re-does its most expensive
work constantly.

---

## Stage 9 — UI polish pass

- Split the layout: **sidebar for inputs, main area for outputs**.
- `st.metric()` row for rows / columns / missing / duplicates.
- Real `st.progress()` bars driven by a `progress_callback` passed into
  the training functions — "3 of 5 done" instead of one opaque spinner.
- Download buttons everywhere: comparison tables as CSV, charts as PNG,
  the winning model as a `.pkl`.
- `.streamlit/config.toml` for consistent theming.

**Design note:** the progress callback lives in `models.py` as a plain
function argument, so the training code stays independent of *how* (or
whether) progress gets displayed.

---

## Stage 10 — ML rigor pass

Three additions that make the numbers more honest:

- **ROC-AUC**, with a `predict_proba` → `decision_function` fallback for
  models like SVC that don't expose probabilities. Unlike F1 it ignores
  where the decision threshold sits, so it measures the model's ranking
  ability rather than its final calls.
- **Cross-validated scores** (CV Mean ± CV Std) next to the single-split
  score. A big gap between them means that headline number would move a
  lot on a different split — far more informative than one number alone.
- **Predict on new data:** upload a second CSV and score it with
  `transform_new_data()`, which reuses the *already-fitted* pipeline.
  Nothing is re-fitted; re-fitting on new data would mean scoring it with
  statistics the model never trained under.

---

## Stage 11 — Two bugs worth remembering

**Bug 1: "I don't see anything on the UI."** Clicking Predict appeared to
do nothing. The code was correct — but its `st.error()` rendered in the
*sidebar*, buried below four other sections, while the user was watching
the *main area* where the results table would appear.

> **Lesson.** Feedback belongs where the user is looking, not where the
> code happens to run. Fixed by storing the outcome (success *or*
> failure) in session state and rendering it in the main area, plus a
> `st.toast()` popup that's hard to miss.

**Bug 2: module changes not taking effect.** Editing `preprocessing.py`
produced an `ImportError` that persisted through reloads.

> **Lesson.** Streamlit re-runs `app.py` but does **not** re-import
> modules already loaded in `sys.modules`. After editing any file other
> than `app.py`, stop the server and restart it. This recurred often
> enough to earn a place in the README.

Also added in this stage: an **analysis summary** under the predictions
table — class distribution, average confidence, a count of low-confidence
rows, and a drift check comparing the predicted class mix against the
training mix.

---

## Stage 12 — Optional EDA extras, and the setuptools saga

**New file: `profiling.py`** — one-line EDA reports from
**ydata-profiling** (the library formerly called pandas-profiling) and
**Sweetviz**.

**Design rule: optional means optional.** Both libraries are heavy, so
the imports are *lazy* — written inside the functions, not at the top of
the file. A missing library then only matters at the moment you ask for
that report, instead of preventing the whole app from starting.
`importlib.util.find_spec()` checks availability *without* importing,
which is what lets the UI label a button "Installed" without paying a
multi-second import cost on every rerun.

**Concept: `components.html()`.** A profiling report is a complete HTML
document with its own CSS and JavaScript. `st.markdown()` can't show it —
it strips scripts, and the styling would collide with the app's. So the
report goes into a sandboxed `<iframe>`, which needs a fixed pixel height
because an iframe can't measure its own contents.

Then came a three-round dependency fight:

1. **ydata-profiling refuses Python 3.14** (`requires-python >=3.10,<3.14`).
   → Installed **Python 3.13.15** alongside, created `venv313`.
2. **`No module named 'pkg_resources'`.** Python 3.12+ stopped seeding
   new virtual environments with setuptools.
   → `pip install setuptools`. Still broken.
3. **Why it was still broken:** setuptools **82.0.0** (Feb 2026) *removed*
   `pkg_resources` entirely, so installing the newest setuptools
   genuinely doesn't contain it — while ydata-profiling still imports it.
   → Pin `setuptools<82`. **81.0.0** is the last version that has it.

> **Lesson.** "Install the package" and "get the module" are not the same
> thing. The app now translates a `ModuleNotFoundError` into the exact
> pip command that fixes it, via a small import-name → pip-name table,
> because "No module named 'pkg_resources'" tells you nothing about
> needing `setuptools<82`.

---

## Stage 13 — The professional restructure

An audit of the codebase against normal engineering standards found the
logic modules in good shape — but one file was the problem.

**`app.py` was 782 lines of top-level script nested 10 levels deep**,
with 234 lines indented past 20 spaces. At that depth nobody can tell
which `if` a line belongs to.

What changed:

| Change | Why |
| --- | --- |
| `app.py`: **782 → 56 lines**, nesting **10 → 1** | The whole shape of the app now fits on one screen |
| Split into `ui_sidebar.py` / `ui_main.py` | Sidebar *writes* session state, main area *reads* it — so every value has exactly one origin |
| New `config.py` | Every magic number in one place, each with its reasoning |
| New `state.py` | Session-state bookkeeping and cached work, collected |
| New `tests/` — **60 tests** | Including an anti-leakage guard and a "does the pipeline actually learn a known signal" check |
| New `README.md` | Quick start, architecture, troubleshooting |
| Renumbered UI sections | They had run 1, 2, 3, 5, 6, 8, 9 — now Step 1–7 |
| "Stage 5:" docstrings rewritten | Build chronology means nothing to a new reader |

Verified by running the 60 tests plus 15 end-to-end UI scenarios against
the refactored code before shipping it.

---

## Final structure

```
automl_app/
├── app.py                    56 lines  — entry point; sets up the page, calls the UI
├── config.py                106 lines  — every tunable number, with reasoning
├── state.py                 160 lines  — session state + cached expensive work
├── ui_sidebar.py            389 lines  — all input controls   (WRITES state)
├── ui_main.py               487 lines  — all output rendering (READS state)
├── preprocessing.py         267 lines  — splitting, imputing, encoding, scaling, selection
├── models.py                344 lines  — model definitions, training, tuning
├── evaluation.py            165 lines  — metrics, comparison table, picking a winner
├── visualization.py         189 lines  — matplotlib figures
├── profiling.py             216 lines  — optional EDA reports (ydata / sweetviz)
├── tests/
│   ├── conftest.py                     — shared fixtures
│   ├── test_preprocessing.py           — 17 tests
│   ├── test_evaluation.py              — 14 tests
│   ├── test_models.py                  — 16 tests
│   └── test_profiling.py               — 13 tests
├── README.md                           — documentation
├── BUILD_JOURNAL.md                    — this file
├── requirements.txt                    — core dependencies
├── requirements-extras.txt             — optional EDA dependencies
├── install_extras.bat                  — one-click environment setup
├── .streamlit/config.toml              — theme
└── .gitignore
```

**Two conventions hold it together.** The five logic modules
(`preprocessing`, `models`, `evaluation`, `visualization`, `profiling`)
import **no Streamlit at all** — they're plain Python that takes data and
returns results, which is what makes them testable without a browser.
And input is separated from output: the sidebar writes session state, the
main area reads it.

---

## Running totals

| | |
| --- | --- |
| Dataset | 54,808 rows × 14 columns |
| Train / test split | 43,846 / 10,962 |
| Models compared | 5 per problem type, baseline + tuned |
| Best result | Random Forest, F1 = 0.9044 |
| Tests | 60, all passing |
| Python | 3.13.15 (`venv313`) |

---

## The recurring lessons

1. **Streamlit re-runs everything, always.** Session state and
   `@st.cache_data` exist because of this, and most bugs trace back to
   forgetting it.
2. **Restart after editing any module but `app.py`.** Python caches
   imports; Streamlit's reload doesn't clear them.
3. **Fit on training data only.** The single rule that separates honest
   scores from flattering ones.
4. **Show feedback where the user is looking.** A correct error message
   in the wrong place is an invisible error message.
5. **New Python versions break data tooling.** 3.13 over 3.14 was worth
   the detour.
6. **Write down why a number is what it is.** Every cap and threshold in
   `config.py` carries its reasoning, so none of them looks arbitrary in
   six months.
