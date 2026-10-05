"""
Tests for evaluation.py - metrics, the comparison table, picking a
winner, and the plain-language summary of a batch of predictions.
"""

import numpy as np
import pandas as pd
import pytest

import config
from evaluation import (
    build_comparison_table,
    evaluate_classification,
    evaluate_regression,
    primary_metric_name,
    select_best_model,
    summarize_predictions,
)


# --- metrics ----------------------------------------------------------
def test_perfect_classification_scores_one():
    y = [0, 1, 0, 1]
    metrics = evaluate_classification(y, y)

    assert metrics["Accuracy"] == 1.0
    assert metrics["F1-Score"] == 1.0


def test_roc_auc_is_only_reported_when_scores_are_supplied():
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 0, 1]

    assert "ROC-AUC" not in evaluate_classification(y_true, y_pred)

    proba = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.1, 0.9]])
    assert evaluate_classification(y_true, y_pred, y_score=proba)["ROC-AUC"] == 1.0


def test_roc_auc_returns_nan_rather_than_raising_on_a_single_class():
    """A test split containing only one class can't have an ROC curve."""
    y_true = [1, 1, 1, 1]
    proba = np.array([[0.1, 0.9]] * 4)

    result = evaluate_classification(y_true, y_true, y_score=proba)
    assert np.isnan(result["ROC-AUC"])


def test_perfect_regression_scores_r2_of_one():
    y = [1.0, 2.0, 3.0]
    metrics = evaluate_regression(y, y)

    assert metrics["R2 Score"] == pytest.approx(1.0)
    assert metrics["MAE"] == pytest.approx(0.0)


def test_rmse_is_the_square_root_of_mse():
    metrics = evaluate_regression([1.0, 2.0, 3.0], [1.5, 2.5, 2.0])
    assert metrics["RMSE"] == pytest.approx(np.sqrt(metrics["MSE"]))


# --- choosing a winner ------------------------------------------------
def test_primary_metric_differs_by_problem_type():
    assert primary_metric_name("classification") == "F1-Score"
    assert primary_metric_name("regression") == "R2 Score"


def test_best_model_is_the_highest_scorer():
    table = build_comparison_table({
        "Model A": {"F1-Score": 0.70},
        "Model B": {"F1-Score": 0.91},
        "Model C": {"F1-Score": 0.85},
    })
    assert select_best_model(table, "classification") == "Model B"


def test_comparison_table_has_one_row_per_model():
    table = build_comparison_table({
        "A": {"Accuracy": 0.5, "F1-Score": 0.4},
        "B": {"Accuracy": 0.6, "F1-Score": 0.7},
    })

    assert list(table.index) == ["A", "B"]
    assert "Accuracy" in table.columns


# --- summarize_predictions --------------------------------------------
def test_classification_summary_counts_each_class():
    preds = np.array(["yes", "yes", "no", "yes"])
    summary = summarize_predictions(preds, "classification")

    assert summary["n"] == 4
    assert summary["top_class"] == "yes"
    assert summary["top_class_share"] == pytest.approx(75.0)
    assert summary["class_counts"]["no"] == 1


def test_confidence_stats_flag_low_confidence_rows():
    preds = np.array([0, 1, 1])
    proba = np.array([[0.95, 0.05], [0.45, 0.55], [0.10, 0.90]])

    summary = summarize_predictions(preds, "classification", proba)

    assert summary["avg_confidence"] == pytest.approx((0.95 + 0.55 + 0.90) / 3)
    assert summary["low_confidence_threshold"] == config.LOW_CONFIDENCE_THRESHOLD
    assert summary["low_confidence_count"] == 1  # only the 0.55 row


def test_confidence_stats_are_absent_without_probabilities():
    summary = summarize_predictions(np.array([0, 1]), "classification")
    assert "avg_confidence" not in summary


def test_regression_summary_describes_the_spread():
    summary = summarize_predictions(np.array([10.0, 20.0, 30.0]), "regression")

    assert summary["mean"] == pytest.approx(20.0)
    assert summary["median"] == pytest.approx(20.0)
    assert summary["min"] == pytest.approx(10.0)
    assert summary["max"] == pytest.approx(30.0)


def test_empty_prediction_batch_is_handled():
    """An empty CSV must report nothing, not divide by zero."""
    assert summarize_predictions(np.array([]), "classification") == {"n": 0}


def test_single_row_regression_batch_has_zero_spread():
    """pandas .std() of one value is NaN; the summary must say 0.0 instead."""
    summary = summarize_predictions(np.array([5.0]), "regression")
    assert summary["std"] == 0.0
