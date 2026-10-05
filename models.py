"""
The models themselves: which ones to try, how to train them, and how to
search for better hyperparameters.

Two entry points:
  train_and_evaluate  - fit every model once with sensible defaults,
                        score it on the test set, and cross-validate it
                        on the training set for a stability check.
  tune_and_evaluate   - search each model's hyperparameter grid with
                        cross-validation, then score the best version
                        found on the same untouched test set.

Five models per problem type, chosen to span genuinely different
approaches - a linear model, a single tree, an ensemble of trees, a
margin-based model and a distance-based one - so the comparison says
something about the data rather than comparing five variations of the
same idea.

Plain Python - no Streamlit here. Progress is reported back to whatever
is calling through an optional progress_callback, which keeps this
module independent of how (or whether) that progress gets displayed.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from config import (
    CV_FOLDS,
    GRID_SEARCH_MAX_COMBOS,
    RANDOM_SEARCH_N_ITER,
    RANDOM_STATE,
    SVM_MAX_TRAIN_ROWS,
)
from evaluation import evaluate_classification, evaluate_regression


def get_models(problem_type: str) -> dict:
    """Return {model_name: unfitted estimator} for the given problem type."""
    if problem_type == "classification":
        return {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
            "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE),
            "SVM": SVC(random_state=RANDOM_STATE),
            "KNN": KNeighborsClassifier(),
        }
    else:
        return {
            "Linear Regression": LinearRegression(),
            "Decision Tree": DecisionTreeRegressor(random_state=RANDOM_STATE),
            "Random Forest": RandomForestRegressor(random_state=RANDOM_STATE),
            "SVR": SVR(),
            "KNN": KNeighborsRegressor(),
        }


def get_param_grids(problem_type: str) -> dict:
    """
    Return {model_name: param_grid} - the hyperparameter values to
    search over for each model. Linear Regression has no grid: with
    no regularization, it has no hyperparameter worth tuning, so it's
    trained once as a baseline instead.

    Random Forest's grid intentionally excludes unlimited tree depth
    (max_depth=None) and caps n_estimators at 150: unbounded, heavily
    forested trees are by far the slowest combination to
    cross-validate, and testing showed capping them cut Random
    Forest's search time from ~120s to ~20s on a ~44k-row training
    set with negligible score difference.
    """
    if problem_type == "classification":
        return {
            "Logistic Regression": {"C": [0.01, 0.1, 1, 10]},
            "Decision Tree": {
                "max_depth": [3, 5, 10, None],
                "min_samples_split": [2, 5, 10],
            },
            "Random Forest": {
                "n_estimators": [100, 150],
                "max_depth": [5, 10],
                "min_samples_split": [2, 5],
            },
            "SVM": {"C": [0.1, 1, 10], "kernel": ["rbf", "linear"]},
            "KNN": {"n_neighbors": [3, 5, 7, 9, 11]},
        }
    else:
        return {
            "Linear Regression": {},
            "Decision Tree": {
                "max_depth": [3, 5, 10, None],
                "min_samples_split": [2, 5, 10],
            },
            "Random Forest": {
                "n_estimators": [100, 150],
                "max_depth": [5, 10],
                "min_samples_split": [2, 5],
            },
            "SVR": {
                "C": [0.1, 1, 10],
                "kernel": ["rbf", "linear"],
                "epsilon": [0.1, 0.2],
            },
            "KNN": {"n_neighbors": [3, 5, 7, 9, 11]},
        }


def _grid_size(param_grid: dict) -> int:
    size = 1
    for values in param_grid.values():
        size *= len(values)
    return size


def _effective_cv(y_fit, problem_type: str, cv: int):
    """
    Cross-validation requires every class to appear at least `cv`
    times (one appearance per fold). With an imbalanced target and a
    small dataset, the rarest class can fall short of that. Rather
    than let GridSearchCV crash, shrink cv down to what the rarest
    class can actually support (minimum 2 folds), or signal that
    tuning must be skipped entirely if even 2 isn't possible.

    Returns (effective_cv_or_None, note_or_None). effective_cv is
    None when there's no safe way to cross-validate at all (skip
    tuning for that model and just fit it with defaults).
    """
    if problem_type != "classification":
        return cv, None

    min_class_count = pd.Series(y_fit).value_counts().min()
    if min_class_count >= cv:
        return cv, None
    if min_class_count >= 2:
        return int(min_class_count), (
            f"Reduced to {min_class_count}-fold CV: the rarest class in "
            f"the training data only has {min_class_count} examples."
        )
    return None, (
        "Skipped tuning: the rarest class has only 1 example, which isn't "
        "enough to cross-validate on. Used default hyperparameters instead."
    )


def _get_decision_scores(model, X):
    """
    For ROC-AUC we need a CONFIDENCE score per prediction, not just
    the final class label. Most classifiers give this via
    predict_proba(); SVC without probability=True doesn't have that
    (enabling it makes fitting much slower, since it runs internal
    cross-validation), but does have decision_function(), which
    works just as well for ranking-based metrics like ROC-AUC.
    Returns None if the model supports neither.
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return None


def _subsample_for_svm(X_train, y_train, random_state: int = RANDOM_STATE):
    """Randomly cap SVM/SVR's training data at SVM_MAX_TRAIN_ROWS rows."""
    n = X_train.shape[0]
    if n <= SVM_MAX_TRAIN_ROWS:
        return X_train, y_train, False

    rng = np.random.default_rng(random_state)
    idx = rng.choice(n, size=SVM_MAX_TRAIN_ROWS, replace=False)

    y_arr = y_train.to_numpy() if hasattr(y_train, "to_numpy") else np.asarray(y_train)
    return X_train[idx], y_arr[idx], True


def train_and_evaluate(models: dict, X_train, X_test, y_train, y_test,
                        problem_type: str, progress_callback=None,
                        cv: int = CV_FOLDS):
    """
    Fit every model once with default hyperparameters and evaluate it
    on the test set. Also computes a CROSS-VALIDATED
    score on the training data (mean +/- std across `cv` folds) for
    the same primary metric - a single train/test split can be
    lucky or unlucky, so seeing how much the score swings across
    different folds is a much more honest signal of how reliable a
    model's number actually is than the single test-set score alone.

    Returns results, fitted_models, predictions, notes (extra info
    to show in the UI).

    progress_callback, if given, is called as
    progress_callback(model_name, done_count, total_count) right
    after each model finishes - lets the UI show real per-model
    progress instead of one opaque "please wait" spinner.
    """
    results, fitted_models, predictions, notes = {}, {}, {}, {}
    total = len(models)
    scoring = "f1_weighted" if problem_type == "classification" else "r2"

    for i, (name, model) in enumerate(models.items(), start=1):
        if name in ("SVM", "SVR"):
            X_fit, y_fit, was_subsampled = _subsample_for_svm(X_train, y_train)
            if was_subsampled:
                notes[name] = (
                    f"Trained on a random {SVM_MAX_TRAIN_ROWS:,}-row subsample "
                    f"of the {X_train.shape[0]:,} training rows (for speed); "
                    f"evaluated on the full test set."
                )
        else:
            X_fit, y_fit = X_train, y_train

        model.fit(X_fit, y_fit)
        y_pred = model.predict(X_test)
        y_score = _get_decision_scores(model, X_test) if problem_type == "classification" else None

        if problem_type == "classification":
            metrics = evaluate_classification(y_test, y_pred, y_score=y_score)
        else:
            metrics = evaluate_regression(y_test, y_pred)

        effective_cv, cv_note = _effective_cv(y_fit, problem_type, cv)
        if effective_cv is not None:
            cv_scores = cross_val_score(model, X_fit, y_fit, cv=effective_cv, scoring=scoring, n_jobs=1)
            metrics["CV Mean"] = cv_scores.mean()
            metrics["CV Std"] = cv_scores.std()
        else:
            metrics["CV Mean"] = np.nan
            metrics["CV Std"] = np.nan
            notes[name] = (notes.get(name, "") + " " + cv_note).strip()

        results[name] = metrics
        fitted_models[name] = model
        predictions[name] = y_pred

        if progress_callback is not None:
            progress_callback(name, i, total)

    return results, fitted_models, predictions, notes


def tune_and_evaluate(problem_type: str, X_train, X_test, y_train, y_test,
                       cv: int = CV_FOLDS, random_state: int = RANDOM_STATE,
                       progress_callback=None):
    """
    For every model, search its hyperparameter grid (Grid Search if
    small enough to be exhaustive, otherwise Random Search)
    using cross-validation on the TRAINING data only, then evaluate
    the best-found version of each model on the untouched test set.

    progress_callback, if given, is called as
    progress_callback(model_name, done_count, total_count) right
    after each model finishes tuning.

    Returns a dict with: tuned_models, best_params, search_notes,
    results, predictions.
    """
    models = get_models(problem_type)
    param_grids = get_param_grids(problem_type)
    scoring = "f1_weighted" if problem_type == "classification" else "r2"

    tuned_models, best_params, search_notes = {}, {}, {}
    results, predictions = {}, {}
    total = len(models)

    for i, (name, model) in enumerate(models.items(), start=1):
        param_grid = param_grids.get(name, {})

        X_fit, y_fit = X_train, y_train
        subsample_note = ""
        if name in ("SVM", "SVR"):
            X_fit, y_fit, was_subsampled = _subsample_for_svm(
                X_train, y_train, random_state=random_state
            )
            if was_subsampled:
                subsample_note = (
                    f" Searched using a random {SVM_MAX_TRAIN_ROWS:,}-row "
                    f"subsample of the {X_train.shape[0]:,} training rows."
                )

        effective_cv, cv_note = _effective_cv(y_fit, problem_type, cv)

        if not param_grid or effective_cv is None:
            model.fit(X_fit, y_fit)
            tuned_models[name] = model
            best_params[name] = {}
            if not param_grid:
                search_notes[name] = "No hyperparameters to tune - trained as a baseline."
            else:
                search_notes[name] = cv_note + subsample_note
        else:
            combos = _grid_size(param_grid)
            if combos <= GRID_SEARCH_MAX_COMBOS:
                search = GridSearchCV(
                    model, param_grid, scoring=scoring, cv=effective_cv, n_jobs=1
                )
                search_notes[name] = (
                    f"Grid Search: tried all {combos} combinations "
                    f"({effective_cv}-fold CV)."
                    f"{' ' + cv_note if cv_note else ''}{subsample_note}"
                )
            else:
                n_iter = min(RANDOM_SEARCH_N_ITER, combos)
                search = RandomizedSearchCV(
                    model, param_grid, n_iter=n_iter, scoring=scoring,
                    cv=effective_cv, n_jobs=1, random_state=random_state,
                )
                search_notes[name] = (
                    f"Random Search: sampled {n_iter} of {combos} possible "
                    f"combinations ({effective_cv}-fold CV)."
                    f"{' ' + cv_note if cv_note else ''}{subsample_note}"
                )

            search.fit(X_fit, y_fit)
            tuned_models[name] = search.best_estimator_
            best_params[name] = search.best_params_

        y_pred = tuned_models[name].predict(X_test)
        y_score = (
            _get_decision_scores(tuned_models[name], X_test)
            if problem_type == "classification" else None
        )
        if problem_type == "classification":
            metrics = evaluate_classification(y_test, y_pred, y_score=y_score)
        else:
            metrics = evaluate_regression(y_test, y_pred)

        results[name] = metrics
        predictions[name] = y_pred

        if progress_callback is not None:
            progress_callback(name, i, total)

    return {
        "tuned_models": tuned_models,
        "best_params": best_params,
        "search_notes": search_notes,
        "results": results,
        "predictions": predictions,
    }
