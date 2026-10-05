"""
Session state and cached loaders - the Streamlit plumbing.

WHY SESSION STATE EXISTS AT ALL: Streamlit re-runs your whole script
from the top every time anyone touches any widget. An ordinary Python
variable therefore dies and is recreated on every interaction, which
would mean a trained model vanishing the moment you moved an unrelated
slider. `st.session_state` is the dictionary that survives those reruns,
so everything expensive or irreplaceable lives in it.

Keeping that bookkeeping here - rather than scattered through the UI
code - means there is exactly one list of what the app remembers, and
one definition of what "start again from here" clears.
"""

from typing import NamedTuple, Optional

import pandas as pd
import streamlit as st

import config
import profiling
from evaluation import build_comparison_table, select_best_model


# Every key the app stores. Declaring them in one tuple means init_state()
# can guarantee each one exists before any code reads it, so the UI never
# has to write `st.session_state.get("thing")` defensively.
STATE_KEYS = (
    "df",                  # the uploaded dataset
    "file_signature",      # (name, size) - detects "a different file was picked"
    "target_col",
    "problem_type",        # "classification" or "regression"
    "preprocess_result",   # dict from preprocessing.preprocess_and_select
    "y_train",
    "y_test",
    "train_results",       # baseline models: results/fitted_models/predictions/notes
    "tune_results",        # same shape, for the hyperparameter-tuned versions
    "split_note",          # warning raised while splitting, if any
    "new_predictions",     # predictions on a second, unseen CSV
    "new_predict_error",
    "eda_report",          # generated profiling report: tool/html/rows_used
    "eda_error",
)


def init_state() -> None:
    """Give every key a value (None) before anything tries to read it."""
    for key in STATE_KEYS:
        if key not in st.session_state:
            st.session_state[key] = None


def clear_all() -> None:
    """Full reset - the 'Start Over' button."""
    for key in STATE_KEYS:
        st.session_state[key] = None


def reset_downstream_state() -> None:
    """
    Clear everything derived from the dataset, keeping the dataset itself.

    Called when the uploaded file or the chosen target changes, because
    at that point every later result describes data or a question that no
    longer applies - stale results on screen are worse than none. The EDA
    report is included because Sweetviz colours its charts by the target,
    so a new target means a genuinely different report.
    """
    for key in STATE_KEYS:
        if key not in ("df", "file_signature"):
            st.session_state[key] = None


def clear_prediction_state() -> None:
    """Drop predictions on new data - they belong to a model that just changed."""
    st.session_state.new_predictions = None
    st.session_state.new_predict_error = None


def clear_model_results() -> None:
    """
    Drop everything produced by training, keeping the preprocessing.

    Used when the data is re-processed: the models that exist were fitted
    on the previous features, so they no longer describe anything real.
    The EDA report deliberately survives - it describes the raw uploaded
    data, which re-processing doesn't change.
    """
    st.session_state.train_results = None
    st.session_state.tune_results = None
    clear_prediction_state()


class BestModel(NamedTuple):
    """
    The model the app currently considers best, and what's needed to talk
    about it. A NamedTuple rather than a plain 5-tuple so call sites read
    `best.name` instead of `best[0]` - same cost, far clearer.
    """
    name: str
    model: object
    predictions: object      # its predictions on the test set
    source: str              # "tuned" or "baseline"
    table: pd.DataFrame      # the comparison table it was chosen from


def get_current_best(problem_type: str) -> Optional[BestModel]:
    """
    Tuned models win if hyperparameter tuning has been run, otherwise the
    baselines. Returns None when nothing has been trained yet.

    Both the Visualizations section and the predict-on-new-data feature
    need "the current best model", so the rule lives here once rather
    than being re-derived (and eventually re-derived differently) in two
    places.
    """
    if st.session_state.tune_results is not None:
        tuned = st.session_state.tune_results
        table = build_comparison_table(tuned["results"])
        name = select_best_model(table, problem_type)
        return BestModel(name, tuned["tuned_models"][name],
                         tuned["predictions"][name], "tuned", table)

    if st.session_state.train_results is not None:
        trained = st.session_state.train_results
        table = build_comparison_table(trained["results"])
        name = select_best_model(table, problem_type)
        return BestModel(name, trained["fitted_models"][name],
                         trained["predictions"][name], "baseline", table)

    return None


# ----------------------------------------------------------------------
# Cached work
# ----------------------------------------------------------------------
# @st.cache_data memoises a function against its ARGUMENTS: call it again
# with the same inputs and Streamlit returns the stored result instead of
# recomputing. Given that every widget interaction re-runs the entire
# script, this is what stops a 3.6MB CSV being re-parsed - and a
# multi-minute profiling report being rebuilt - on every single click.

@st.cache_data(show_spinner="Reading CSV...")
def load_csv(file) -> pd.DataFrame:
    """Parse an uploaded CSV once per upload, not once per rerun."""
    return pd.read_csv(file)


@st.cache_data(show_spinner=False, max_entries=config.EDA_CACHE_MAX_ENTRIES)
def build_ydata_report(df: pd.DataFrame, minimal: bool) -> str:
    """Cached ydata-profiling HTML. See config.EDA_CACHE_MAX_ENTRIES."""
    return profiling.build_ydata_profile_html(df, minimal=minimal)


@st.cache_data(show_spinner=False, max_entries=config.EDA_CACHE_MAX_ENTRIES)
def build_sweetviz_report(df: pd.DataFrame, target_col: Optional[str]) -> str:
    """Cached Sweetviz HTML."""
    return profiling.build_sweetviz_html(df, target_col=target_col)
