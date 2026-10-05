"""
The main area: everything the app SHOWS.

Nothing here reads a widget or decides anything - every function takes
what it needs from session state and renders it. That one-way flow
(sidebar writes, main area reads) is what keeps a Streamlit app from
turning into a tangle, because a value can only ever be set in one
place.

render_main() lists the sections in page order; each one is a small
function below it.
"""

import io

import joblib
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

import config
import state
import visualization as viz
from evaluation import (
    build_comparison_table,
    primary_metric_name,
    select_best_model,
    summarize_predictions,
)


def render_main() -> None:
    st.title("🤖 AutoML App")

    df = st.session_state.df
    if df is None:
        st.info("Upload a CSV in the sidebar to get started.")
        return

    _dataset_overview(df)
    _target_summary()

    if st.session_state.preprocess_result is None:
        if st.session_state.target_col is not None:
            st.info("Click **Process** in the sidebar to detect the problem "
                    "type and preprocess.")
    elif st.session_state.train_results is None:
        _preprocessing_summary()
        st.info("Click **Train Models** in the sidebar to train and compare models.")
    else:
        _preprocessing_summary()
        _model_results(df)
        _new_data_predictions()

    _eda_report()


# ----------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------
def _dataset_overview(df: pd.DataFrame) -> None:
    st.success(f"Loaded dataset: {df.shape[0]:,} rows x {df.shape[1]} columns.")

    row_count, col_count, missing, duplicates = st.columns(4)
    row_count.metric("Rows", f"{df.shape[0]:,}")
    col_count.metric("Columns", df.shape[1])
    missing.metric("Missing values", f"{int(df.isna().sum().sum()):,}")
    duplicates.metric("Duplicate rows", f"{int(df.duplicated().sum()):,}")

    st.subheader("Preview (first 5 rows)")
    st.dataframe(df.head(), use_container_width=True)

    with st.expander("Dataset overview (dtypes & missing values per column)"):
        st.dataframe(
            pd.DataFrame({
                "dtype": df.dtypes.astype(str),
                "missing": df.isna().sum(),
                "missing %": (df.isna().mean() * 100).round(1),
                "unique values": df.nunique(),
            }),
            use_container_width=True,
        )


def _target_summary() -> None:
    if st.session_state.problem_type is None:
        return

    st.write(f"Target column: **{st.session_state.target_col}**")
    st.write(f"Detected problem type: **{st.session_state.problem_type.title()}**")
    if st.session_state.split_note:
        st.warning(st.session_state.split_note)


# ----------------------------------------------------------------------
# Preprocessing
# ----------------------------------------------------------------------
def _preprocessing_summary() -> None:
    result = st.session_state.preprocess_result

    st.subheader("Preprocessing summary")

    train_rows, test_rows, features = st.columns(3)
    train_rows.metric("Training rows", f"{result['X_train'].shape[0]:,}")
    test_rows.metric("Test rows", f"{result['X_test'].shape[0]:,}")
    features.metric("Features selected", len(result["selected_feature_names"]))

    st.write(
        f"Detected **{len(result['numeric_cols'])}** numeric columns "
        f"(missing values filled with the median, then scaled) and "
        f"**{len(result['categorical_cols'])}** categorical columns "
        f"(missing values filled with the most frequent value, then "
        f"one-hot encoded)."
    )
    st.write(
        f"That produced **{len(result['all_feature_names'])}** engineered "
        f"columns in total, narrowed down to the "
        f"**{len(result['selected_feature_names'])}** most predictive "
        f"via SelectKBest."
    )

    if result["dropped_empty_columns"]:
        st.warning(
            "Dropped entirely-empty column(s) (100% missing values): "
            + ", ".join(result["dropped_empty_columns"])
        )

    with st.expander("Show selected feature names"):
        st.write(result["selected_feature_names"])


# ----------------------------------------------------------------------
# Model results
# ----------------------------------------------------------------------
def _model_results(df: pd.DataFrame) -> None:
    problem_type = st.session_state.problem_type
    metric = primary_metric_name(problem_type)

    baseline_table, baseline_best = _baseline_comparison(problem_type, metric)

    if st.session_state.tune_results is not None:
        _tuned_comparison(problem_type, metric, baseline_table, baseline_best)

    best = state.get_current_best(problem_type)
    _best_model_download(best)
    _visualizations(df, best, problem_type, metric)


def _baseline_comparison(problem_type: str, metric: str):
    trained = st.session_state.train_results
    table = build_comparison_table(trained["results"])
    best_name = select_best_model(table, problem_type)

    st.subheader("Baseline model comparison")
    st.caption(
        f"CV Mean / CV Std: the same metric, cross-validated on the training "
        f"data ({config.CV_FOLDS} folds). Compare it to the single-split "
        f"{metric} to judge how stable each model's score really is - a big "
        f"gap means that number would move a lot on a different split."
    )
    st.dataframe(_style_table(table, metric), use_container_width=True)

    st.download_button(
        "Download baseline comparison (CSV)",
        data=table.to_csv().encode("utf-8"),
        file_name="baseline_comparison.csv",
        mime="text/csv",
    )
    st.success(
        f"Best baseline model: **{best_name}** "
        f"({metric} = {table.loc[best_name, metric]:.4f})"
    )

    for name, note in trained["notes"].items():
        if note:
            st.caption(f"Note on {name}: {note}")

    return table, best_name


def _tuned_comparison(problem_type: str, metric: str,
                       baseline_table: pd.DataFrame, baseline_best: str) -> None:
    tuned = st.session_state.tune_results
    table = build_comparison_table(tuned["results"])
    best_name = select_best_model(table, problem_type)

    st.subheader("Tuned model comparison")
    st.dataframe(_style_table(table, metric), use_container_width=True)

    st.download_button(
        "Download tuned comparison (CSV)",
        data=table.to_csv().encode("utf-8"),
        file_name="tuned_comparison.csv",
        mime="text/csv",
    )

    improvement = (table.loc[best_name, metric]
                   - baseline_table.loc[baseline_best, metric])
    st.success(
        f"Best tuned model: **{best_name}** "
        f"({metric} = {table.loc[best_name, metric]:.4f}, "
        f"{'+' if improvement >= 0 else ''}{improvement:.4f} vs. best baseline)"
    )

    with st.expander("Show search details and best hyperparameters per model"):
        for name in tuned["tuned_models"]:
            st.write(f"**{name}**")
            st.caption(tuned["search_notes"][name])
            st.write(tuned["best_params"][name] or "_(no tunable hyperparameters)_")


def _best_model_download(best: state.BestModel) -> None:
    """
    Serialise the winning model to a .pkl the user can load elsewhere
    with joblib.load(). io.BytesIO() is an in-memory file: joblib writes
    bytes into it exactly as it would to disk, and those bytes go
    straight to the download button - no temporary file involved.
    """
    buffer = io.BytesIO()
    joblib.dump(best.model, buffer)

    st.download_button(
        f"Download best model - {best.name} (.pkl)",
        data=buffer.getvalue(),
        file_name=f"best_model_{best.name.replace(' ', '_').lower()}.pkl",
        mime="application/octet-stream",
    )


# ----------------------------------------------------------------------
# Charts
# ----------------------------------------------------------------------
def _visualizations(df: pd.DataFrame, best: state.BestModel,
                     problem_type: str, metric: str) -> None:
    st.subheader("Visualizations")
    st.caption(f"Charts for the current best model: **{best.name}** ({best.source}).")

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    tabs = st.tabs([
        "Feature Distributions", "Correlation Heatmap", "Model Comparison",
        "Feature Importance", "Predictions",
    ])

    with tabs[0]:
        _chart_or_message(
            viz.plot_feature_distributions(df, numeric_cols),
            "feature_distributions.png", "dl_dist",
            "No numeric columns to plot.",
        )

    with tabs[1]:
        _chart_or_message(
            viz.plot_correlation_heatmap(df, numeric_cols),
            "correlation_heatmap.png", "dl_corr",
            "Need at least 2 numeric columns for a correlation heatmap.",
        )

    with tabs[2]:
        _chart_or_message(
            viz.plot_model_comparison(best.table, metric),
            "model_comparison.png", "dl_compare",
        )

    with tabs[3]:
        _chart_or_message(
            viz.plot_feature_importance(
                best.model, st.session_state.preprocess_result["selected_feature_names"]
            ),
            "feature_importance.png", "dl_importance",
            f"{best.name} doesn't expose feature importances or coefficients "
            f"(KNN and SVM with a non-linear kernel have no such notion).",
        )

    with tabs[4]:
        if problem_type == "classification":
            figure = viz.plot_confusion_matrix(st.session_state.y_test, best.predictions)
            filename = "confusion_matrix.png"
        else:
            figure = viz.plot_predicted_vs_actual(st.session_state.y_test, best.predictions)
            filename = "predicted_vs_actual.png"
        _chart_or_message(figure, filename, "dl_pred")


def _chart_or_message(figure, filename: str, key: str,
                       empty_message: str = "Nothing to plot.") -> None:
    """
    Render a chart with its own PNG download, or explain why there isn't
    one. Every plotting function returns None when the chart doesn't
    apply, so this single helper handles the "or not" case for all of
    them instead of repeating the same if/else five times.
    """
    if figure is None:
        st.info(empty_message)
        return

    st.pyplot(figure)
    st.download_button(
        "Download chart (PNG)",
        data=viz.fig_to_png_bytes(figure),
        file_name=filename,
        mime="image/png",
        key=key,
    )


# ----------------------------------------------------------------------
# Predictions on new data
# ----------------------------------------------------------------------
def _new_data_predictions() -> None:
    predictions = st.session_state.new_predictions
    error = st.session_state.new_predict_error

    # Rendered whenever a prediction has been ATTEMPTED, so the result is
    # visible either way - a silent failure is the worst outcome.
    if predictions is None and error is None:
        return

    st.subheader("Predictions on new data")

    if error is not None:
        st.error(error)
        return

    st.caption(f"Predicted using: {predictions['model_name']}")

    output = predictions["new_df"].copy()
    output["Prediction"] = predictions["preds"]
    if predictions["proba"] is not None:
        output["Confidence"] = predictions["proba"].max(axis=1).round(4)

    st.dataframe(output, use_container_width=True)
    st.download_button(
        "Download predictions (CSV)",
        data=output.to_csv(index=False).encode("utf-8"),
        file_name="predictions.csv",
        mime="text/csv",
    )

    _prediction_analysis(predictions)


def _prediction_analysis(predictions: dict) -> None:
    """
    The table above is the raw answer; this turns it into the few numbers
    someone actually wants - what did the model mostly say, how sure was
    it, and does this batch even look like the data it learned from?
    """
    problem_type = st.session_state.problem_type
    summary = summarize_predictions(
        predictions["preds"], problem_type, predictions["proba"]
    )
    if summary["n"] == 0:
        return

    st.markdown("**Analysis of this batch**")

    if problem_type == "classification":
        _classification_analysis(summary)
    else:
        _regression_analysis(summary)


def _classification_analysis(summary: dict) -> None:
    has_confidence = "avg_confidence" in summary

    columns = st.columns(3 if has_confidence else 2)
    columns[0].metric("Records predicted", f"{summary['n']:,}")
    columns[1].metric(
        f"Most common: {summary['top_class']}", f"{summary['top_class_share']:.1f}%"
    )
    if has_confidence:
        columns[2].metric("Avg. confidence", f"{summary['avg_confidence']:.1%}")

    if has_confidence:
        confidence_note = (
            f"The model's average confidence in its own top prediction was "
            f"**{summary['avg_confidence']:.1%}**, and "
            f"**{summary['low_confidence_count']}** record(s) fell below the "
            f"{summary['low_confidence_threshold']:.0%} mark - worth a closer "
            f"look before acting on those."
        )
    else:
        confidence_note = (
            "This model doesn't expose prediction probabilities, so no "
            "confidence score is available."
        )

    st.write(
        f"Out of **{summary['n']:,}** record(s), "
        f"**{summary['top_class_share']:.1f}%** were predicted as "
        f"**{summary['top_class']}**, the most common outcome in this batch. "
        f"{confidence_note}"
    )

    _class_distribution_table(summary)


def _class_distribution_table(summary: dict) -> None:
    """Predicted class mix, next to the training mix it should resemble."""
    distribution = pd.DataFrame({
        "Predicted count": summary["class_counts"],
        "Predicted %": summary["class_percentages"],
    })

    y_train = st.session_state.y_train
    if y_train is not None:
        training_share = (y_train.value_counts(normalize=True) * 100).round(1)
        distribution["Training %"] = (
            training_share.reindex(distribution.index).fillna(0.0)
        )
        drift = (distribution["Predicted %"] - distribution["Training %"]).abs().max()

        if drift >= config.CLASS_DRIFT_WARNING_PP:
            st.warning(
                f"This batch's predicted class mix differs from the training "
                f"data by up to {drift:.1f} percentage points on at least one "
                f"class - the new data may look meaningfully different from "
                f"what the model was trained on."
            )

    st.dataframe(distribution, use_container_width=True)


def _regression_analysis(summary: dict) -> None:
    mean, median, spread, span = st.columns(4)
    mean.metric("Mean", f"{summary['mean']:.4g}")
    median.metric("Median", f"{summary['median']:.4g}")
    spread.metric("Std. dev", f"{summary['std']:.4g}")
    span.metric("Range", f"{summary['min']:.4g} - {summary['max']:.4g}")

    st.write(
        f"The **{summary['n']:,}** predicted value(s) average "
        f"**{summary['mean']:.4g}**, with a spread (std. dev) of "
        f"**{summary['std']:.4g}**, ranging from **{summary['min']:.4g}** "
        f"to **{summary['max']:.4g}**."
    )


# ----------------------------------------------------------------------
# Automated EDA report
# ----------------------------------------------------------------------
def _eda_report() -> None:
    report = st.session_state.eda_report
    error = st.session_state.eda_error

    if report is None and error is None:
        return

    st.subheader("Automated EDA report")

    if error is not None:
        st.error(error)
        return

    scope = (
        f"random sample of {report['rows_used']:,} rows"
        if report["was_sampled"] else f"all {report['rows_used']:,} rows"
    )
    st.caption(f"Generated by {report['tool']} ({scope}).")

    # Offered BEFORE the embedded copy: these reports are built to be
    # opened standalone in a browser, which reads far better than a
    # scrolling frame.
    st.download_button(
        "Download full report (HTML)",
        data=report["html"],
        file_name=f"eda_report_{report['tool'].replace('-', '_').lower()}.html",
        mime="text/html",
    )

    # components.html() drops the report into a sandboxed <iframe>, so
    # its own CSS and JavaScript run in isolation instead of colliding
    # with the app's styling. st.markdown() can't do this - it strips
    # scripts. The cost of an iframe is that it can't measure its own
    # contents, hence a fixed height and internal scrolling.
    components.html(report["html"], height=config.EDA_IFRAME_HEIGHT, scrolling=True)


# ----------------------------------------------------------------------
# Shared helper
# ----------------------------------------------------------------------
def _style_table(table: pd.DataFrame, metric: str):
    """Four decimal places everywhere, winning score highlighted."""
    return table.style.highlight_max(
        subset=[metric], color=config.CHART_HIGHLIGHT_COLOR
    ).format("{:.4f}", na_rep="—")
