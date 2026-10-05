"""
Every chart the app draws.

One rule holds throughout: each function BUILDS and RETURNS a matplotlib
Figure, and never calls plt.show(). There is no screen to show a window
on - the app is a web page - so the caller passes the returned figure to
st.pyplot() instead. That also means these functions can be tested, or
reused in a notebook or a script, with no Streamlit involved.

Functions return None when a chart simply doesn't apply (no numeric
columns to plot, a model with no notion of feature importance) so the
caller can say why rather than showing an empty frame.
"""

import io

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # render to an image buffer; no GUI window anywhere
import matplotlib.pyplot as plt  # noqa: E402  (must follow matplotlib.use)
from sklearn.metrics import confusion_matrix

import config


def fig_to_png_bytes(fig, dpi: int = config.PNG_EXPORT_DPI) -> bytes:
    """Render a matplotlib Figure to PNG bytes, for st.download_button."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    return buf.getvalue()


def _clean_name(name: str) -> str:
    """'num__age' -> 'age', 'cat__region_region_5' -> 'region_region_5'."""
    return name.split("__", 1)[-1] if "__" in name else name


def plot_feature_distributions(df: pd.DataFrame, numeric_cols: list,
                                max_plots: int = config.MAX_DISTRIBUTION_PLOTS):
    """Grid of histograms, one per numeric column (capped so the page doesn't explode)."""
    cols_to_plot = numeric_cols[:max_plots]
    n = len(cols_to_plot)
    if n == 0:
        return None

    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = np.array(axes).reshape(-1)

    for i, col in enumerate(cols_to_plot):
        axes[i].hist(df[col].dropna(), bins=30, color=config.CHART_PRIMARY_COLOR, edgecolor="white")
        axes[i].set_title(col, fontsize=10)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Feature distributions", fontsize=13)
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, numeric_cols: list):
    """Correlation matrix of the numeric columns, including the target if numeric."""
    if len(numeric_cols) < 2:
        return None

    corr = df[numeric_cols].corr()
    size = max(4, 0.6 * len(numeric_cols))
    fig, ax = plt.subplots(figsize=(size + 1, size))

    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(numeric_cols)))
    ax.set_xticklabels(numeric_cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(numeric_cols)))
    ax.set_yticklabels(numeric_cols, fontsize=8)

    for i in range(len(numeric_cols)):
        for j in range(len(numeric_cols)):
            value = corr.iloc[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7,
                     color="white" if abs(value) > 0.5 else "black")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Correlation heatmap")
    fig.tight_layout()
    return fig


def plot_model_comparison(comparison_table: pd.DataFrame, metric: str):
    """Horizontal bar chart ranking every model by one metric, best highlighted."""
    fig, ax = plt.subplots(figsize=(8, max(2.5, 0.6 * len(comparison_table))))

    values = comparison_table[metric].sort_values()
    best_name = values.idxmax()
    colors = [config.CHART_HIGHLIGHT_COLOR if name == best_name else config.CHART_PRIMARY_COLOR for name in values.index]

    ax.barh(values.index, values.values, color=colors)
    for i, v in enumerate(values.values):
        ax.text(v, i, f" {v:.4f}", va="center", fontsize=9)

    ax.set_xlabel(metric)
    ax.set_title(f"Model comparison by {metric}")
    fig.tight_layout()
    return fig


def plot_feature_importance(model, feature_names: list,
                             top_n: int = config.TOP_N_FEATURE_IMPORTANCE):
    """
    Works for tree-based models (feature_importances_) and linear
    models (coef_). Returns None for model types with neither
    (e.g. KNN, or SVM with a non-linear kernel) - not every model
    exposes an interpretable notion of "importance".
    """
    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_)
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_)
        importances = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)
    else:
        return None

    clean_names = [_clean_name(n) for n in feature_names]
    order = np.argsort(importances)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(8, max(3, 0.35 * len(order))))
    ax.barh([clean_names[i] for i in order][::-1], importances[order][::-1], color=config.CHART_PRIMARY_COLOR)
    ax.set_xlabel("Importance")
    ax.set_title("Feature importance")
    fig.tight_layout()
    return fig


def plot_predicted_vs_actual(y_test, y_pred):
    """Scatter of predicted vs. actual values for a regression model, with a
    dashed diagonal marking perfect predictions - points closer to that line
    are better predictions."""
    y_test_arr = np.asarray(y_test, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test_arr, y_pred_arr, alpha=0.3, s=12, color=config.CHART_PRIMARY_COLOR)

    lo = min(y_test_arr.min(), y_pred_arr.min())
    hi = max(y_test_arr.max(), y_pred_arr.max())
    ax.plot([lo, hi], [lo, hi], "r--", label="Perfect prediction")

    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Predicted vs. Actual")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_confusion_matrix(y_test, y_pred):
    """Heatmap of a classification model's confusion matrix: rows are the
    true class, columns are the predicted class, so the diagonal is every
    correct prediction and everything off it is a mistake."""
    y_test_arr = np.asarray(y_test)
    y_pred_arr = np.asarray(y_pred)
    labels = np.unique(np.concatenate([y_test_arr, y_pred_arr]))

    cm = confusion_matrix(y_test_arr, y_pred_arr, labels=labels)
    size = max(4, 0.8 * len(labels))
    fig, ax = plt.subplots(figsize=(size + 1, size))

    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    threshold = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > threshold else "black")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Confusion matrix")
    fig.tight_layout()
    return fig
