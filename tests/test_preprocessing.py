"""
Tests for preprocessing.py.

Each test name says what behaviour is being pinned down, so a failure
report reads like a sentence about what broke. The tests here matter
more than most: preprocessing is where data leakage would silently creep
in, and leakage doesn't announce itself - it just quietly inflates every
score downstream.
"""

import numpy as np
import pandas as pd
import pytest

from preprocessing import (
    DataValidationError,
    detect_problem_type,
    preprocess_and_select,
    split_data,
    transform_new_data,
)


# --- detect_problem_type ----------------------------------------------
def test_binary_numeric_target_is_classification(classification_df):
    assert detect_problem_type(classification_df, "target") == "classification"


def test_continuous_numeric_target_is_regression(regression_df):
    assert detect_problem_type(regression_df, "price") == "regression"


def test_text_target_is_always_classification(classification_df):
    # You can't average "north" and "south", so a non-numeric column can
    # only ever be a set of labels.
    assert detect_problem_type(classification_df, "region") == "classification"


def test_constant_target_is_rejected():
    df = pd.DataFrame({"feature": [1, 2, 3], "target": [7, 7, 7]})
    with pytest.raises(DataValidationError, match="one distinct value"):
        detect_problem_type(df, "target")


def test_all_missing_target_is_rejected():
    df = pd.DataFrame({"feature": [1, 2, 3], "target": [np.nan] * 3})
    with pytest.raises(DataValidationError, match="no non-missing values"):
        detect_problem_type(df, "target")


# --- split_data -------------------------------------------------------
def test_split_separates_target_and_keeps_every_row(classification_df):
    X_train, X_test, y_train, y_test, note = split_data(
        classification_df, "target", problem_type="classification"
    )

    assert "target" not in X_train.columns
    assert len(X_train) + len(X_test) == len(classification_df)
    assert len(y_train) == len(X_train)
    assert note is None


def test_split_is_stratified_for_classification(classification_df):
    """Both sides of the split should keep roughly the original class mix."""
    _, _, y_train, y_test, _ = split_data(
        classification_df, "target", problem_type="classification"
    )
    overall = classification_df["target"].mean()

    assert y_train.mean() == pytest.approx(overall, abs=0.05)
    assert y_test.mean() == pytest.approx(overall, abs=0.05)


def test_split_falls_back_when_a_class_is_too_rare_to_stratify():
    """
    Stratifying needs at least 2 examples per class (one for each side).
    With a single-member class the split must still succeed, and say why
    it couldn't stratify, rather than raising.
    """
    df = pd.DataFrame({
        "feature": range(20),
        "target": [0] * 19 + [1],  # class "1" appears exactly once
    })

    _, _, _, _, note = split_data(df, "target", problem_type="classification")

    assert note is not None
    assert "stratified" in note.lower()


def test_single_column_dataset_is_rejected():
    df = pd.DataFrame({"target": [1, 0, 1, 0]})
    with pytest.raises(DataValidationError, match="one column"):
        split_data(df, "target")


# --- preprocess_and_select --------------------------------------------
def test_preprocessing_produces_numeric_arrays_with_no_missing_values(classification_df):
    X_train, X_test, y_train, _, _ = split_data(
        classification_df, "target", problem_type="classification"
    )
    result = preprocess_and_select(X_train, X_test, y_train, "classification", k_best=5)

    assert result["X_train"].shape[0] == len(X_train)
    assert result["X_test"].shape[0] == len(X_test)
    # Imputation must have dealt with `with_gaps` entirely.
    assert not np.isnan(result["X_train"]).any()
    assert not np.isnan(result["X_test"]).any()


def test_feature_selection_keeps_exactly_k_features(classification_df):
    X_train, X_test, y_train, _, _ = split_data(
        classification_df, "target", problem_type="classification"
    )
    result = preprocess_and_select(X_train, X_test, y_train, "classification", k_best=4)

    assert result["X_train"].shape[1] == 4
    assert len(result["selected_feature_names"]) == 4


def test_k_best_larger_than_available_columns_is_capped(classification_df):
    """Asking for 500 features from a dataset that has ~8 must not raise."""
    X_train, X_test, y_train, _, _ = split_data(
        classification_df, "target", problem_type="classification"
    )
    result = preprocess_and_select(X_train, X_test, y_train, "classification", k_best=500)

    assert result["X_train"].shape[1] == len(result["all_feature_names"])


def test_entirely_empty_columns_are_dropped_and_reported(classification_df):
    classification_df["all_missing"] = np.nan
    X_train, X_test, y_train, _, _ = split_data(
        classification_df, "target", problem_type="classification"
    )
    result = preprocess_and_select(X_train, X_test, y_train, "classification", k_best=5)

    assert "all_missing" in result["dropped_empty_columns"]


def test_scaling_is_fitted_on_training_data_only(classification_df):
    """
    The anti-leakage test. A scaler fitted on the training set leaves THAT
    set centred near mean 0 - the test set is merely transformed with the
    training set's statistics, so it is free to differ. If someone ever
    refits on the full data, training-set centring stays perfect while the
    guarantee this test encodes quietly stops being meaningful.
    """
    X_train, X_test, y_train, _, _ = split_data(
        classification_df, "target", problem_type="classification"
    )
    result = preprocess_and_select(X_train, X_test, y_train, "classification", k_best=5)

    numeric_means = result["X_train"].mean(axis=0)
    assert np.abs(numeric_means).max() < 0.5


# --- transform_new_data -----------------------------------------------
def test_new_data_is_transformed_to_the_same_shape(classification_df):
    X_train, X_test, y_train, _, _ = split_data(
        classification_df, "target", problem_type="classification"
    )
    result = preprocess_and_select(X_train, X_test, y_train, "classification", k_best=5)

    # New data arrives WITHOUT the target column, exactly as in real use.
    new_df = classification_df.drop(columns=["target"]).head(10)
    X_new = transform_new_data(new_df, result)

    assert X_new.shape == (10, result["X_train"].shape[1])


def test_new_data_missing_a_feature_column_is_rejected_by_name(classification_df):
    X_train, X_test, y_train, _, _ = split_data(
        classification_df, "target", problem_type="classification"
    )
    result = preprocess_and_select(X_train, X_test, y_train, "classification", k_best=5)

    incomplete = classification_df.drop(columns=["target", "score"])

    with pytest.raises(DataValidationError, match="score"):
        transform_new_data(incomplete, result)


def test_new_data_may_carry_extra_columns(classification_df):
    """Unexpected extra columns are ignored, not treated as an error."""
    X_train, X_test, y_train, _, _ = split_data(
        classification_df, "target", problem_type="classification"
    )
    result = preprocess_and_select(X_train, X_test, y_train, "classification", k_best=5)

    new_df = classification_df.drop(columns=["target"]).head(5).copy()
    new_df["unrelated_note"] = "ignore me"

    assert transform_new_data(new_df, result).shape[0] == 5
