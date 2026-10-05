"""
Everything that happens to the data before a model ever sees it.

Four jobs, in order:
  1. decide whether the chosen target is a classification or a
     regression problem (detect_problem_type),
  2. split the data into training and test sets (split_data),
  3. fill missing values, encode categories, scale numbers and keep only
     the most predictive features (preprocess_and_select),
  4. apply that exact same fitted pipeline to brand new data later
     (transform_new_data).

Steps 3 and 4 are deliberately separate: everything is FITTED on the
training data only, then merely APPLIED everywhere else. Fitting on data
you later evaluate against leaks information from it into the model and
makes the scores a flattering fiction.

Functions raise DataValidationError for problems a user can fix (a
constant target, an all-empty column) so the app can show a plain
sentence instead of a traceback.

Plain Python - no Streamlit here, so it's all testable directly.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config


class DataValidationError(ValueError):
    """
    Raised for a data problem the user needs to fix (wrong target
    column, not enough data, etc.) - as opposed to a bug. app.py
    catches this specifically and shows the message directly with
    st.error() instead of a raw traceback.
    """
    pass


def detect_problem_type(
    df: pd.DataFrame,
    target_col: str,
    max_unique_for_classification: int = config.MAX_UNIQUE_FOR_CLASSIFICATION,
    unique_ratio_threshold: float = config.UNIQUE_RATIO_THRESHOLD,
) -> str:
    """
    Decide whether predicting `target_col` is a classification or
    regression problem, using simple, explainable rules:

    1. Non-numeric dtype (object, category, bool) -> always classification.
       (You can't average "red", "blue", "green".)
    2. Numeric dtype -> classification if the column has few distinct
       values relative to the dataset (few unique values usually means
       discrete categories/labels encoded as numbers, e.g. 0/1 flags,
       1-5 star ratings), otherwise regression.

    Returns "classification" or "regression".
    """
    target = df[target_col].dropna()

    if len(target) == 0:
        raise DataValidationError(
            f"'{target_col}' has no non-missing values - there's nothing to predict."
        )

    n_unique = target.nunique()
    if n_unique < 2:
        raise DataValidationError(
            f"'{target_col}' only has one distinct value ({target.iloc[0]!r}) - "
            f"a model can't learn to predict a column that never changes. "
            f"Pick a different target column."
        )

    if not pd.api.types.is_numeric_dtype(target):
        return "classification"

    n_rows = len(target)
    unique_ratio = n_unique / n_rows if n_rows > 0 else 0

    if n_unique <= max_unique_for_classification or unique_ratio < unique_ratio_threshold:
        return "classification"

    return "regression"


def split_data(df: pd.DataFrame, target_col: str,
                test_size: float = config.TEST_SIZE,
                random_state: int = config.RANDOM_STATE,
                problem_type: str = "classification"):
    """
    Separate features (X) from the target (y), then split into
    train/test sets.

    For classification we "stratify" on y - this keeps the same
    proportion of each class in both the train and test sets. Without
    it, a random split of an imbalanced target (e.g. 90% "not
    promoted", 10% "promoted") could by chance dump most of the rare
    class into just one of the two sets. Stratifying requires every
    class to have at least 2 members (one for each side of the
    split); if some rare class has only 1 example, we fall back to a
    plain (non-stratified) split rather than crashing, and report
    that in the returned note.

    Returns: X_train, X_test, y_train, y_test, note (str or None)
    """
    if len(df.columns) < 2:
        raise DataValidationError(
            "The dataset only has one column, so there are no feature "
            "columns left once the target is removed."
        )

    X = df.drop(columns=[target_col])
    y = df[target_col]

    note = None
    stratify = None
    if problem_type == "classification":
        min_class_count = y.value_counts().min()
        if min_class_count >= 2:
            stratify = y
        else:
            note = (
                "Skipped stratified splitting: the rarest class in the "
                "target has only 1 example, so the train/test split isn't "
                "guaranteed to have the same class balance in both sets."
            )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )
    return X_train, X_test, y_train, y_test, note


def build_preprocessor(X: pd.DataFrame) -> tuple[ColumnTransformer, list, list]:
    """
    Build (but don't fit) a ColumnTransformer that:
      - for NUMERIC columns: fills missing values with the column
        median, then scales to mean 0 / std 1 (StandardScaler).
      - for CATEGORICAL columns: fills missing values with the most
        frequent value, then one-hot encodes (each category becomes
        its own 0/1 column).

    Returns the transformer plus the numeric and categorical column
    lists it was built from (handy for displaying a summary later).
    """
    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = X.select_dtypes(exclude="number").columns.tolist()

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols),
    ])

    return preprocessor, numeric_cols, categorical_cols


def preprocess_and_select(X_train: pd.DataFrame, X_test: pd.DataFrame,
                           y_train: pd.Series, problem_type: str,
                           k_best: int = config.K_BEST_DEFAULT):
    """
    Fit the preprocessor on X_train only, transform both X_train and
    X_test with it, then run feature selection (SelectKBest) to keep
    only the k_best most predictive columns - again, fit using only
    the training data and y_train.

    Columns that are ENTIRELY missing in X_train are dropped first
    (there's nothing to impute them from - scikit-learn would
    otherwise silently drop them mid-pipeline with just a console
    warning the user would never see).

    SelectKBest scores every feature against the target using a
    statistical test - f_classif (ANOVA F-test) for classification,
    f_regression (correlation-based F-test) for regression - and
    keeps the top-scoring k. This both speeds up training and
    reduces overfitting from irrelevant/noisy columns.

    Returns a dict: X_train, X_test, preprocessor, selector,
    numeric_cols, categorical_cols, all_feature_names,
    selected_feature_names, dropped_empty_columns.
    """
    empty_cols = X_train.columns[X_train.isna().all()].tolist()
    if empty_cols:
        X_train = X_train.drop(columns=empty_cols)
        X_test = X_test.drop(columns=empty_cols)

    if X_train.shape[1] == 0:
        raise DataValidationError(
            "Every feature column is entirely missing - there's no usable "
            "data left to train on."
        )

    preprocessor, numeric_cols, categorical_cols = build_preprocessor(X_train)

    X_train_pre = preprocessor.fit_transform(X_train)
    X_test_pre = preprocessor.transform(X_test)

    all_feature_names = preprocessor.get_feature_names_out()

    # Never ask for more features than actually exist after encoding.
    k_actual = min(k_best, X_train_pre.shape[1])

    score_func = f_classif if problem_type == "classification" else f_regression
    selector = SelectKBest(score_func=score_func, k=k_actual)

    X_train_selected = selector.fit_transform(X_train_pre, y_train)
    X_test_selected = selector.transform(X_test_pre)

    selected_mask = selector.get_support()
    selected_feature_names = all_feature_names[selected_mask].tolist()

    return {
        "X_train": X_train_selected,
        "X_test": X_test_selected,
        "preprocessor": preprocessor,
        "selector": selector,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "all_feature_names": all_feature_names.tolist(),
        "selected_feature_names": selected_feature_names,
        "dropped_empty_columns": empty_cols,
    }


def transform_new_data(new_df: pd.DataFrame, preprocess_result: dict):
    """
    Apply an ALREADY-FITTED preprocessing pipeline (from
    preprocess_and_select) to brand new data for prediction - the
    same imputation values, the same one-hot categories, the same
    scaling statistics, and the same selected features as training.
    Nothing here is re-fit: fitting on new data would mean the model
    was evaluated on statistics it was never trained with, which
    defeats the point of testing generalization.

    Raises DataValidationError if the new data is missing a column
    the pipeline was trained on. Extra/unexpected columns in new_df
    are simply ignored.
    """
    required_cols = preprocess_result["numeric_cols"] + preprocess_result["categorical_cols"]
    missing = [c for c in required_cols if c not in new_df.columns]
    if missing:
        raise DataValidationError(
            "The new data is missing column(s) the model was trained on: "
            + ", ".join(missing)
        )

    X_new = new_df[required_cols]
    X_new_pre = preprocess_result["preprocessor"].transform(X_new)
    X_new_selected = preprocess_result["selector"].transform(X_new_pre)
    return X_new_selected
