"""
Every tunable number in the app, in one place.

Why this file exists: a value like `0.6` sitting in the middle of a
function tells you nothing, and when the same number appears in three
files you can never be sure you changed all of them. Giving each one a
name and a home turns "what does 0.6 mean here?" into
LOW_CONFIDENCE_THRESHOLD, and turns "change the sample size" into a
one-line edit instead of a search across the codebase.

This module imports nothing, so anything may safely import it.
"""

# --- Reproducibility --------------------------------------------------
# One seed shared by every random operation (train/test splitting, SVM
# subsampling, Random Search, EDA sampling). Fixing it means two runs on
# the same data give the same answer - which is what makes results
# comparable between runs, and demos repeatable.
RANDOM_STATE = 42


# --- Train / test split -----------------------------------------------
TEST_SIZE = 0.2  # 20% held back, never seen during training


# --- Problem-type detection -------------------------------------------
# A numeric target is treated as CLASSIFICATION rather than regression
# when it has few distinct values - either fewer than this many
# outright, or fewer than this fraction of the row count. The intuition:
# a column of 0/1 flags or 1-5 star ratings is a set of labels that
# happen to be stored as numbers, not a continuous quantity.
MAX_UNIQUE_FOR_CLASSIFICATION = 20
UNIQUE_RATIO_THRESHOLD = 0.05


# --- Feature selection (SelectKBest slider) ---------------------------
K_BEST_DEFAULT = 20
K_BEST_MIN = 5
K_BEST_MAX = 50
K_BEST_STEP = 5


# --- Model training ---------------------------------------------------
# SVMs scale roughly quadratically (or worse) with the number of training
# rows, so on a dataset of tens of thousands of rows a single SVC/SVR fit
# - let alone a cross-validated hyperparameter search over one - can take
# a very long time. SVM/SVR is therefore always FITTED on a random
# subsample capped at this size. It is still EVALUATED on the full,
# untouched test set; only its training data is capped.
SVM_MAX_TRAIN_ROWS = 5000

# Hyperparameter search sizing. An exhaustive Grid Search is used while a
# grid is small enough to try every combination; past that the app
# switches to Random Search, which samples this many random combinations
# instead. Far cheaper, and on a large grid it lands on results almost as
# good as checking all of them.
GRID_SEARCH_MAX_COMBOS = 20
RANDOM_SEARCH_N_ITER = 15

# Folds for cross-validation. 3 rather than the more common 5 or 10
# because every extra fold means refitting every model again, and this
# app trains ten of them (five baseline, five tuned) while a person
# waits.
CV_FOLDS = 3


# --- Reading the prediction output ------------------------------------
# A prediction whose winning class scores below this is flagged as one
# worth a human look, rather than being acted on automatically.
LOW_CONFIDENCE_THRESHOLD = 0.6

# If the predicted class mix on new data differs from the training class
# mix by more than this many PERCENTAGE POINTS on any class, the app says
# so: the new data may not resemble what the model learned from, which is
# the everyday form of "distribution shift".
CLASS_DRIFT_WARNING_PP = 15


# --- Automated EDA reports (the optional extras) ----------------------
# Profiling libraries scan every column many times over, so they get slow
# on big datasets. Analysing a random sample gives a report that looks
# essentially identical - distributions and correlations are statistical
# properties, so a few thousand rows capture them faithfully.
EDA_SAMPLE_DEFAULT = 5000
EDA_SAMPLE_MIN = 1000
EDA_SAMPLE_MAX = 20000

# Below this many rows there is no point offering a sample slider at all.
EDA_SAMPLE_SLIDER_THRESHOLD = 1000

# An <iframe> cannot measure its own contents, so an embedded report
# needs a fixed pixel height and scrolls internally.
EDA_IFRAME_HEIGHT = 800

# These reports are multi-megabyte HTML strings; cap how many are held in
# Streamlit's cache at once so memory doesn't creep up over a session.
EDA_CACHE_MAX_ENTRIES = 3


# --- Charts -----------------------------------------------------------
CHART_PRIMARY_COLOR = "#4C72B0"
CHART_HIGHLIGHT_COLOR = "#2e7d32"  # marks the winning model / best score
MAX_DISTRIBUTION_PLOTS = 12  # a histogram per column, capped so the page stays readable
TOP_N_FEATURE_IMPORTANCE = 15
PNG_EXPORT_DPI = 150
