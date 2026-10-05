"""
The sidebar: every input and every button in the app.

The split between this file and ui_main.py follows one rule - the
sidebar takes input and WRITES to session state, the main area READS
session state and renders output. Keeping that boundary means you can
answer "where does this value come from?" by looking in one file, and
"where is this shown?" by looking in the other.

Each step is its own small function called in order by render_sidebar(),
so the whole workflow is readable at a glance at the top of the file,
and no function is more than a screenful.
"""

import pandas as pd
import streamlit as st

import config
import profiling
import state
from models import get_models, train_and_evaluate, tune_and_evaluate
from preprocessing import (
    DataValidationError,
    detect_problem_type,
    preprocess_and_select,
    split_data,
    transform_new_data,
)


def render_sidebar() -> None:
    """
    Draw the sidebar. The nesting here mirrors what the user can
    actually do: you can't train before preprocessing, and you can't
    tune or predict before training, so those controls simply don't
    appear yet. Automated EDA sits outside that chain because it only
    ever needed the raw dataframe.
    """
    with st.sidebar:
        st.header("AutoML Controls")

        _dataset_step()
        df = st.session_state.df
        if df is None:
            return

        _target_step(df)
        _preprocess_step(df)

        if st.session_state.preprocess_result is not None:
            _train_step()

        if st.session_state.train_results is not None:
            _tune_step()
            _predict_step()

        _eda_step(df)


# ----------------------------------------------------------------------
# Step 1 - load a dataset
# ----------------------------------------------------------------------
def _dataset_step() -> None:
    st.subheader("Step 1 · Dataset")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if st.button("Start Over", use_container_width=True):
        state.clear_all()
        st.rerun()

    if uploaded_file is None:
        return

    # (name, size) distinguishes "the user picked a different file" from
    # "the script re-ran for some unrelated reason, and the uploader is
    # simply still holding the same file".
    signature = (uploaded_file.name, uploaded_file.size)
    if signature == st.session_state.file_signature:
        return

    try:
        st.session_state.df = state.load_csv(uploaded_file)
    except Exception as e:
        st.error(f"Could not read this file as a CSV: {e}")
        st.session_state.df = None
    else:
        st.session_state.file_signature = signature
        state.reset_downstream_state()


# ----------------------------------------------------------------------
# Step 2 - choose what to predict
# ----------------------------------------------------------------------
def _target_step(df: pd.DataFrame) -> None:
    st.subheader("Step 2 · Target variable")
    target_col = st.selectbox(
        "Column to predict", options=df.columns, label_visibility="collapsed",
    )

    # Changing the target invalidates every result below it.
    if target_col != st.session_state.target_col:
        state.reset_downstream_state()
    st.session_state.target_col = target_col


# ----------------------------------------------------------------------
# Step 3 - preprocess
# ----------------------------------------------------------------------
def _preprocess_step(df: pd.DataFrame) -> None:
    st.subheader("Step 3 · Preprocess")
    k_best = st.slider(
        "Features to keep (selection)",
        min_value=config.K_BEST_MIN,
        max_value=config.K_BEST_MAX,
        value=config.K_BEST_DEFAULT,
        step=config.K_BEST_STEP,
    )

    if not st.button("Process", use_container_width=True, type="primary"):
        return

    target_col = st.session_state.target_col
    try:
        problem_type = detect_problem_type(df, target_col)
        X_train, X_test, y_train, y_test, split_note = split_data(
            df, target_col, problem_type=problem_type
        )
        result = preprocess_and_select(
            X_train, X_test, y_train, problem_type, k_best=k_best
        )
    except DataValidationError as e:
        # Something wrong with the DATA that the user can fix - show the
        # message as written, no traceback.
        st.error(str(e))
    except Exception as e:
        st.error(f"Something went wrong while processing: {e}")
    else:
        # Models trained on the PREVIOUS feature set are meaningless now,
        # so they go; any EDA report stays, since it describes the raw
        # upload, which re-processing doesn't touch.
        state.clear_model_results()
        st.session_state.problem_type = problem_type
        st.session_state.preprocess_result = result
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test
        st.session_state.split_note = split_note


# ----------------------------------------------------------------------
# Step 4 - train the baseline models
# ----------------------------------------------------------------------
def _train_step() -> None:
    st.subheader("Step 4 · Train models")
    if not st.button("Train Models", use_container_width=True, type="primary"):
        return

    result = st.session_state.preprocess_result
    problem_type = st.session_state.problem_type

    try:
        bar, callback = _progress_reporter("Training")
        results, fitted_models, predictions, notes = train_and_evaluate(
            get_models(problem_type),
            result["X_train"], result["X_test"],
            st.session_state.y_train, st.session_state.y_test,
            problem_type, progress_callback=callback,
        )
        bar.empty()
    except Exception as e:
        st.error(f"Training failed: {e}")
    else:
        st.session_state.train_results = {
            "results": results,
            "fitted_models": fitted_models,
            "predictions": predictions,
            "notes": notes,
        }
        st.session_state.tune_results = None
        state.clear_prediction_state()


# ----------------------------------------------------------------------
# Step 5 - tune hyperparameters
# ----------------------------------------------------------------------
def _tune_step() -> None:
    st.subheader("Step 5 · Tune hyperparameters")
    if not st.button("Tune Hyperparameters", use_container_width=True, type="primary"):
        return

    result = st.session_state.preprocess_result

    try:
        bar, callback = _progress_reporter("Tuning")
        tuned = tune_and_evaluate(
            st.session_state.problem_type,
            result["X_train"], result["X_test"],
            st.session_state.y_train, st.session_state.y_test,
            progress_callback=callback,
        )
        bar.empty()
    except Exception as e:
        st.error(f"Hyperparameter tuning failed: {e}")
    else:
        st.session_state.tune_results = tuned
        state.clear_prediction_state()


# ----------------------------------------------------------------------
# Step 6 - predict on a second, unseen CSV
# ----------------------------------------------------------------------
def _predict_step() -> None:
    st.subheader("Step 6 · Predict on new data")
    st.caption("Upload a CSV with the same feature columns (no target needed).")

    new_file = st.file_uploader(
        "New data", type=["csv"], key="predict_upload",
        label_visibility="collapsed",
    )

    if not st.button("Predict", use_container_width=True, type="primary"):
        return

    if new_file is None:
        st.session_state.new_predictions = None
        st.session_state.new_predict_error = "Upload a CSV above first."
    else:
        _run_prediction(new_file)

    # The outcome itself is rendered in the MAIN area, next to where the
    # results table appears - a red box in the sidebar is easy to miss
    # when you're watching the page. The toast is the attention-getter;
    # it only points at the real message.
    if st.session_state.new_predict_error is not None:
        st.toast("Prediction failed - see 'Predictions on new data'.", icon="⚠️")
    else:
        st.toast("Predictions ready - scroll to 'Predictions on new data'.", icon="✅")


def _run_prediction(new_file) -> None:
    """Score an uploaded CSV with the current best model."""
    problem_type = st.session_state.problem_type
    preprocess_result = st.session_state.preprocess_result

    try:
        new_df = pd.read_csv(new_file)
        best = state.get_current_best(problem_type)

        # The SAME fitted pipeline as training: same imputation values,
        # same one-hot categories, same scaling, same selected features.
        # Nothing is re-fitted - fitting on new data would mean scoring
        # it with statistics the model was never trained on.
        X_new = transform_new_data(new_df, preprocess_result)
        predictions = best.model.predict(X_new)

        probabilities = (
            best.model.predict_proba(X_new)
            if problem_type == "classification"
            and hasattr(best.model, "predict_proba")
            else None
        )
    except DataValidationError as e:
        st.session_state.new_predictions = None
        st.session_state.new_predict_error = str(e)
    except Exception as e:
        st.session_state.new_predictions = None
        st.session_state.new_predict_error = f"Prediction failed: {e}"
    else:
        st.session_state.new_predictions = {
            "new_df": new_df,
            "preds": predictions,
            "proba": probabilities,
            "model_name": best.name,
        }
        st.session_state.new_predict_error = None


# ----------------------------------------------------------------------
# Step 7 - automated EDA (optional extras)
# ----------------------------------------------------------------------
def _eda_step(df: pd.DataFrame) -> None:
    """
    Available as soon as a CSV is loaded - these reports describe the raw
    data, so they don't depend on preprocessing or training at all.
    """
    st.subheader("Step 7 · Automated EDA (optional)")

    tool = st.radio(
        "Report type",
        options=list(profiling.PACKAGES),
        captions=[
            "Installed" if profiling.is_installed(info["import_name"])
            else "Not installed"
            for info in profiling.PACKAGES.values()
        ],
        label_visibility="collapsed",
    )

    sample_rows = _eda_sample_control(df)

    # "minimal" skips the pairwise correlation and interaction plots,
    # whose cost grows with the SQUARE of the column count.
    minimal = True
    if tool == "ydata-profiling":
        minimal = not st.checkbox(
            "Full report (slower: adds correlations & interactions)", value=False,
        )

    if st.button("Generate EDA report", use_container_width=True):
        _generate_eda_report(df, tool, sample_rows, minimal)


def _eda_sample_control(df: pd.DataFrame) -> int:
    """A slider only makes sense when there's a range worth choosing from."""
    if len(df) <= config.EDA_SAMPLE_SLIDER_THRESHOLD:
        st.caption(f"Small dataset - analysing all {len(df):,} rows.")
        return len(df)

    sample_cap = min(len(df), config.EDA_SAMPLE_MAX)
    return st.slider(
        "Rows to analyse (random sample)",
        min_value=config.EDA_SAMPLE_MIN,
        max_value=sample_cap,
        value=min(config.EDA_SAMPLE_DEFAULT, sample_cap),
        step=1000,
    )


def _generate_eda_report(df: pd.DataFrame, tool: str,
                          sample_rows: int, minimal: bool) -> None:
    import_name = profiling.PACKAGES[tool]["import_name"]

    if not profiling.is_installed(import_name):
        st.session_state.eda_report = None
        st.session_state.eda_error = (
            f"{tool} isn't installed in this environment. "
            f"Install it with:  {profiling.install_hint(tool)}"
        )
        st.toast("EDA report failed - see 'Automated EDA report'.", icon="⚠️")
        return

    sample_df, was_sampled = profiling.maybe_sample(df, sample_rows)

    try:
        with st.spinner(f"Building the {tool} report..."):
            if tool == "ydata-profiling":
                html = state.build_ydata_report(sample_df, minimal)
            else:
                html = state.build_sweetviz_report(
                    sample_df, st.session_state.target_col
                )
    except Exception as e:
        # A missing dependency deep inside these libraries surfaces as a
        # cryptic module name; translate it into the command that fixes it.
        message = f"Report generation failed: {e}"
        hint = profiling.explain_import_failure(e)
        if hint:
            message += f"\n\n{hint}"
        st.session_state.eda_report = None
        st.session_state.eda_error = message
        st.toast("EDA report failed - see 'Automated EDA report'.", icon="⚠️")
    else:
        st.session_state.eda_report = {
            "tool": tool,
            "html": html,
            "rows_used": len(sample_df),
            "was_sampled": was_sampled,
        }
        st.session_state.eda_error = None
        st.toast("EDA report ready - scroll to 'Automated EDA report'.", icon="✅")


# ----------------------------------------------------------------------
# Shared helper
# ----------------------------------------------------------------------
def _progress_reporter(label: str):
    """
    Build a real progress bar plus the callback that advances it.

    models.train_and_evaluate / tune_and_evaluate call the returned
    callback after each model finishes, so the user sees "3 of 5 done"
    rather than staring at one opaque spinner for two minutes.
    """
    bar = st.progress(0.0, text=f"{label}...")

    def callback(name, done, total):
        bar.progress(done / total, text=f"{label}: {name} done ({done}/{total})")

    return bar, callback
