"""
Tests for models.py.

These run real scikit-learn training, but on the tiny synthetic datasets
from conftest.py, so the whole file still finishes in seconds. The point
isn't to check that scikit-learn works - it's to check that this app
wires it up correctly and survives the awkward cases.
"""

import numpy as np
import pandas as pd
import pytest

from models import (
    _effective_cv,
    _grid_size,
    _subsample_for_svm,
    get_models,
    get_param_grids,
    train_and_evaluate,
    tune_and_evaluate,
)
from preprocessing import preprocess_and_select, split_data


@pytest.fixture
def prepared_classification(classification_df):
    """Run the real preprocessing pipeline, so tests use realistic inputs."""
    X_train, X_test, y_train, y_test, _ = split_data(
        classification_df, "target", problem_type="classification"
    )
    result = preprocess_and_select(X_train, X_test, y_train, "classification", k_best=5)
    return result["X_train"], result["X_test"], y_train, y_test


# --- model definitions ------------------------------------------------
def test_five_models_are_offered_for_each_problem_type():
    assert len(get_models("classification")) == 5
    assert len(get_models("regression")) == 5


def test_every_model_has_a_matching_parameter_grid():
    """A model with no grid entry would be silently skipped during tuning."""
    for problem_type in ("classification", "regression"):
        assert set(get_models(problem_type)) == set(get_param_grids(problem_type))


def test_regression_offers_the_required_model_types():
    names = set(get_models("regression"))
    assert names == {"Linear Regression", "Decision Tree", "Random Forest", "SVR", "KNN"}


def test_grid_size_multiplies_the_options():
    assert _grid_size({"a": [1, 2, 3], "b": [10, 20]}) == 6
    assert _grid_size({}) == 1


# --- cross-validation safety -----------------------------------------
def test_cv_is_unchanged_when_every_class_is_common_enough():
    y = pd.Series([0, 1] * 20)
    folds, note = _effective_cv(y, "classification", cv=3)

    assert folds == 3
    assert note is None


def test_cv_shrinks_to_fit_the_rarest_class():
    y = pd.Series([0] * 20 + [1, 1])  # only two examples of class 1
    folds, note = _effective_cv(y, "classification", cv=5)

    assert folds == 2
    assert "2-fold" in note


def test_cv_is_abandoned_when_a_class_has_a_single_example():
    y = pd.Series([0] * 20 + [1])
    folds, note = _effective_cv(y, "classification", cv=3)

    assert folds is None
    assert "Skipped tuning" in note


def test_regression_never_has_a_class_balance_problem():
    y = pd.Series(np.linspace(0, 100, 50))
    assert _effective_cv(y, "regression", cv=3) == (3, None)


# --- SVM subsampling --------------------------------------------------
def test_small_training_sets_are_not_subsampled():
    X = np.zeros((100, 3))
    y = pd.Series(np.zeros(100))
    X_out, y_out, was_subsampled = _subsample_for_svm(X, y)

    assert was_subsampled is False
    assert X_out.shape[0] == 100


def test_subsampling_is_reproducible_for_a_fixed_seed():
    """Two runs on the same data must pick the same rows, or results drift."""
    X = np.arange(20000).reshape(-1, 2)
    y = pd.Series(np.arange(10000))

    first, _, was_subsampled = _subsample_for_svm(X, y, random_state=42)
    second, _, _ = _subsample_for_svm(X, y, random_state=42)

    assert was_subsampled is True
    assert np.array_equal(first, second)


# --- training ---------------------------------------------------------
def test_training_returns_a_result_for_every_model(prepared_classification):
    X_train, X_test, y_train, y_test = prepared_classification
    models = get_models("classification")

    results, fitted, predictions, _ = train_and_evaluate(
        models, X_train, X_test, y_train, y_test, "classification"
    )

    assert set(results) == set(models)
    assert set(fitted) == set(models)
    for name in models:
        assert len(predictions[name]) == len(y_test)


def test_training_reports_cross_validated_stability(prepared_classification):
    X_train, X_test, y_train, y_test = prepared_classification

    results, _, _, _ = train_and_evaluate(
        get_models("classification"), X_train, X_test, y_train, y_test, "classification"
    )

    for metrics in results.values():
        assert "CV Mean" in metrics
        assert "CV Std" in metrics


def test_progress_callback_fires_once_per_model(prepared_classification):
    X_train, X_test, y_train, y_test = prepared_classification
    seen = []

    train_and_evaluate(
        get_models("classification"), X_train, X_test, y_train, y_test,
        "classification",
        progress_callback=lambda name, done, total: seen.append((done, total)),
    )

    assert seen == [(1, 5), (2, 5), (3, 5), (4, 5), (5, 5)]


def test_a_learnable_dataset_actually_gets_learned(prepared_classification):
    """
    Sanity check on the whole pipeline: `target` is defined by `score`, so
    a model that scores no better than a coin flip means something
    upstream is broken - not that the model is bad.
    """
    X_train, X_test, y_train, y_test = prepared_classification

    results, _, _, _ = train_and_evaluate(
        get_models("classification"), X_train, X_test, y_train, y_test, "classification"
    )

    assert max(m["Accuracy"] for m in results.values()) > 0.8


# --- tuning -----------------------------------------------------------
def test_tuning_returns_a_tuned_model_and_notes_for_each(prepared_classification):
    X_train, X_test, y_train, y_test = prepared_classification

    tuned = tune_and_evaluate("classification", X_train, X_test, y_train, y_test)

    expected = set(get_models("classification"))
    assert set(tuned["tuned_models"]) == expected
    assert set(tuned["search_notes"]) == expected
    assert set(tuned["results"]) == expected


def test_linear_regression_is_reported_as_having_nothing_to_tune(regression_df):
    X_train, X_test, y_train, y_test, _ = split_data(
        regression_df, "price", problem_type="regression"
    )
    prepared = preprocess_and_select(X_train, X_test, y_train, "regression", k_best=4)

    tuned = tune_and_evaluate(
        "regression", prepared["X_train"], prepared["X_test"], y_train, y_test
    )

    assert tuned["best_params"]["Linear Regression"] == {}
    assert "No hyperparameters" in tuned["search_notes"]["Linear Regression"]
