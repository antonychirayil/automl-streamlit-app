"""
Scoring models and choosing a winner.

Turns a set of predictions into metrics, collects those metrics into a
comparison table across models, names the best one, and summarises a
batch of predictions on new data in plain numbers.

Plain Python - no Streamlit here, so every function below can be tested
directly (see tests/test_evaluation.py).
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

import config

# The metric each problem type is judged on when the app has to pick a
# single winner. F1 balances precision against recall, which matters on
# an imbalanced target where plain accuracy flatters a model that just
# predicts the majority class. R2 is the share of the target's variance
# the model explains.
PRIMARY_METRIC = {
    "classification": "F1-Score",
    "regression": "R2 Score",
}


def primary_metric_name(problem_type: str) -> str:
    """The metric this problem type is ranked by."""
    return PRIMARY_METRIC[problem_type]


def evaluate_classification(y_test, y_pred, y_score=None) -> dict:
    """
    Accuracy is the fraction of predictions that were exactly right.

    Precision, recall and F1 use average="weighted", which weights each
    class's score by how many true examples it has - the right choice on
    an imbalanced target, since it neither lets a rare class dominate the
    average nor lets it vanish from it.

    ROC-AUC (only when y_score is given) is the probability that the
    model ranks a random positive example above a random negative one:
    1.0 is perfect ranking, 0.5 is guessing. Unlike the others it ignores
    where the decision threshold happens to sit, so it measures the
    model's underlying confidence rather than its final yes/no calls.
    y_score is per-class probabilities from predict_proba, or a
    decision_function score for models that have no predict_proba.
    """
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "F1-Score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    }

    if y_score is not None:
        metrics["ROC-AUC"] = _safe_roc_auc(y_test, y_score)

    return metrics


def _safe_roc_auc(y_test, y_score) -> float:
    """
    ROC-AUC has real edge cases - a class present in y_test but never
    scored, a single-class test split - and none of them are worth
    crashing a whole training run over. Returns NaN instead, which the
    comparison table renders as a dash.
    """
    try:
        n_classes = len(np.unique(y_test))
        if n_classes == 2:
            # predict_proba returns (n, 2): take the positive class
            # column. decision_function returns a flat (n,): use it as is.
            score = y_score[:, 1] if np.ndim(y_score) == 2 else y_score
            return roc_auc_score(y_test, score)
        return roc_auc_score(y_test, y_score, multi_class="ovr", average="weighted")
    except Exception:
        return np.nan


def evaluate_regression(y_test, y_pred) -> dict:
    """
    MAE  - average absolute error, in the target's own units.
    MSE  - average SQUARED error, so big misses hurt much more than small ones.
    RMSE - the square root of MSE: back in the target's units and easier
           to read, while still penalising large errors heavily.
    R2   - the share of the target's variance the model explains. 1.0 is
           perfect, 0.0 is no better than always guessing the mean, and
           it goes negative for a model that is worse than that.
    """
    mse = mean_squared_error(y_test, y_pred)
    return {
        "MAE": mean_absolute_error(y_test, y_pred),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "R2 Score": r2_score(y_test, y_pred),
    }


def build_comparison_table(results: dict) -> pd.DataFrame:
    """{model_name: {metric: value}} -> one row per model, one column per metric."""
    return pd.DataFrame(results).T


def select_best_model(comparison_table: pd.DataFrame, problem_type: str) -> str:
    """Name of the highest-scoring model on this problem type's primary metric."""
    return comparison_table[primary_metric_name(problem_type)].idxmax()


def summarize_predictions(preds, problem_type: str, proba=None) -> dict:
    """
    Condense a batch of predictions on new data into the handful of
    numbers a person actually wants after clicking Predict, rather than
    making them read every row of the output table.

    Classification: how many records landed in each class, which class
    dominated, and - when the model exposes probabilities - the average
    confidence plus a count of predictions below
    config.LOW_CONFIDENCE_THRESHOLD that deserve a second look.

    Regression: the five-number spread of the predicted values, which
    shows at a glance whether this batch looks plausible or oddly skewed.
    """
    preds = pd.Series(preds)
    count = len(preds)
    if count == 0:
        return {"n": 0}

    summary = {"n": count}

    if problem_type == "classification":
        counts = preds.value_counts()
        summary["class_counts"] = counts
        summary["class_percentages"] = (counts / count * 100).round(1)
        summary["top_class"] = counts.idxmax()
        summary["top_class_share"] = float(counts.max() / count * 100)

        if proba is not None:
            confidence = pd.Series(proba.max(axis=1))
            summary["avg_confidence"] = float(confidence.mean())
            summary["min_confidence"] = float(confidence.min())
            summary["low_confidence_threshold"] = config.LOW_CONFIDENCE_THRESHOLD
            summary["low_confidence_count"] = int(
                (confidence < config.LOW_CONFIDENCE_THRESHOLD).sum()
            )
    else:
        summary["mean"] = float(preds.mean())
        summary["median"] = float(preds.median())
        summary["std"] = float(preds.std()) if count > 1 else 0.0
        summary["min"] = float(preds.min())
        summary["max"] = float(preds.max())

    return summary
