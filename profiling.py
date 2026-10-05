"""
Extras module: one-line automated EDA reports.

This wires in two popular "point it at a dataframe and get a full EDA
report" libraries:

  - ydata-profiling  (this is the library that used to be called
    pandas-profiling; it was renamed in 2023, so `pip install
    pandas-profiling` now just tells you to install this instead)
  - sweetviz

Both do the same job in different styles: they walk every column and
produce one big standalone HTML page - distributions, missing values,
duplicates, correlations, warnings about skew/high cardinality, and so
on. ydata-profiling is the more exhaustive/statistical of the two;
sweetviz is prettier and can highlight how each feature relates to the
target.

--------------------------------------------------------------------
THE ONE DESIGN RULE IN THIS FILE: these are OPTIONAL.
--------------------------------------------------------------------
Both libraries are heavy (ydata-profiling alone pulls in ~15
dependencies), and ydata-profiling currently refuses to install on
Python 3.14 at all. So the app must work perfectly when neither is
installed. Two habits make that true:

  1. LAZY IMPORTS. The `import ydata_profiling` / `import sweetviz`
     lines live INSIDE the functions, not at the top of the file. A
     missing library then only matters at the moment you actually ask
     for that report - importing this module never fails, so the rest
     of the app is unaffected.
  2. is_installed() checks WITHOUT importing, so the UI can grey out a
     button without paying the (slow) import cost just to find out.

Note also that nothing in this file imports streamlit. It's plain
Python that takes a DataFrame and returns an HTML string, which means
you could test it in a normal script or notebook. The Streamlit-specific
parts (caching, buttons, rendering) stay in app.py.
"""

import importlib.util
import tempfile
from pathlib import Path

import pandas as pd

# Pip install names differ from import names, so keep both: the import
# name is what find_spec() looks for, the pip name is what we tell the
# user to install.
PACKAGES = {
    "ydata-profiling": {"import_name": "ydata_profiling", "pip_name": "ydata-profiling"},
    "Sweetviz": {"import_name": "sweetviz", "pip_name": "sweetviz"},
}


def is_installed(import_name: str) -> bool:
    """
    "Is this library available?" without importing it.

    importlib.util.find_spec() only LOOKS for the module (walks the
    same search path `import` would) and returns None if it isn't
    there. That matters here because actually importing
    ydata_profiling takes several seconds - far too slow to do on
    every Streamlit rerun just to decide whether to enable a button.
    """
    try:
        return importlib.util.find_spec(import_name) is not None
    except (ImportError, ValueError):
        # A broken/partial install can raise instead of returning None.
        return False


def install_hint(label: str) -> str:
    """The pip command to show the user when a library is missing."""
    pip_name = PACKAGES[label]["pip_name"]
    return f"pip install {pip_name}"


# Some modules are imported under one name but installed under another,
# so a raw "No module named X" tells you nothing about what to pip
# install. The values here are full pip SPECS, not just names, because
# sometimes the version matters as much as the package.
#
# pkg_resources is the cautionary tale. Two separate things broke it:
#   1. Python 3.12 stopped seeding new virtual environments with
#      setuptools (it used to arrive automatically next to pip).
#   2. setuptools 82.0.0 (Feb 2026) then DELETED pkg_resources
#      outright - it's been deprecated for years in favour of
#      importlib.resources / importlib.metadata.
# So on a fresh 3.13 venv a plain `pip install setuptools` installs
# 82+ and STILL leaves you with no pkg_resources. Pinning below 82 is
# what actually fixes it, until ydata-profiling drops the import.
MODULE_TO_PIP_SPEC = {
    "pkg_resources": '"setuptools<82"',
    "PIL": "pillow",
    "sklearn": "scikit-learn",
    "yaml": "PyYAML",
    "cv2": "opencv-python",
}


def explain_import_failure(exc: Exception) -> str | None:
    """
    Turn a bare ModuleNotFoundError into something the user can act on.

    These report builders pull in long dependency chains, so when one
    link is missing the error surfaces as a cryptic "No module named
    'pkg_resources'" from deep inside a library the user never called.
    This translates that into the exact pip command that fixes it, and
    returns None for any other kind of error so the caller falls back
    to showing the original message.
    """
    if not isinstance(exc, ModuleNotFoundError) or not exc.name:
        return None

    pip_spec = MODULE_TO_PIP_SPEC.get(exc.name, exc.name)
    return (
        f"A library this report depends on is missing: `{exc.name}`. "
        f"Fix it by running `pip install {pip_spec}` in this app's "
        f"environment (the one Streamlit is running from), then restart "
        f"Streamlit."
    )


def maybe_sample(df: pd.DataFrame, sample_rows: int | None = None,
                  random_state: int = 42) -> tuple[pd.DataFrame, bool]:
    """
    Both report generators scan every column many times over, so they
    get slow on big datasets (ydata-profiling on 50k+ rows can take
    minutes). A random sample gives you a report that looks
    essentially the same - distributions and correlations are
    statistical properties, so a few thousand rows captures them
    faithfully - in a fraction of the time.

    Returns (dataframe_to_use, was_sampled).
    """
    if sample_rows is not None and len(df) > sample_rows:
        return df.sample(n=sample_rows, random_state=random_state), True
    return df, False


def build_ydata_profile_html(df: pd.DataFrame, title: str = "Dataset profile",
                              minimal: bool = True) -> str:
    """
    Build a ydata-profiling report and return it as an HTML string.

    minimal=True is the big speed lever: it skips the most expensive
    work (all the pairwise correlation matrices and feature
    interaction plots, which scale with the SQUARE of the column
    count). You still get per-column stats, distributions, missing
    values and the data-quality warnings. Turn it off when you want
    the full statistical treatment and are willing to wait.

    progress_bar=False stops the library writing tqdm progress bars to
    the terminal, which are meaningless here since the user is looking
    at a browser, not the console.
    """
    from ydata_profiling import ProfileReport  # lazy - see module docstring

    report = ProfileReport(df, title=title, minimal=minimal, progress_bar=False)
    return report.to_html()


def is_valid_sweetviz_target(df: pd.DataFrame, target_col: str | None) -> bool:
    """
    Sweetviz can colour every chart by how it relates to a "target
    feature", which is genuinely useful - but it only supports
    numeric or boolean targets, and it errors out if the target has
    any missing values. Rather than let that raise mid-report, we
    check first and simply generate a report without target analysis
    when the target isn't eligible.
    """
    if target_col is None or target_col not in df.columns:
        return False

    column = df[target_col]
    if column.isna().any():
        return False

    return pd.api.types.is_numeric_dtype(column) or pd.api.types.is_bool_dtype(column)


def build_sweetviz_html(df: pd.DataFrame, target_col: str | None = None) -> str:
    """
    Build a Sweetviz report and return it as an HTML string.

    Unlike ydata-profiling (which hands you HTML directly via
    to_html()), Sweetviz only knows how to WRITE its report to a file.
    So we hand it a throwaway file inside a temporary directory, then
    read the text back out. tempfile.TemporaryDirectory() cleans the
    directory up automatically when the `with` block ends, so nothing
    is left lying around on disk.

    open_browser=False is important: without it Sweetviz tries to pop
    open a browser tab on the machine RUNNING the app - harmless on
    your laptop, but wrong in principle (and useless if this were
    deployed to a server).

    layout="vertical" stacks the report in one tall column, which
    suits the scrollable, height-limited frame we embed it in far
    better than the default side-by-side "widescreen" layout.
    """
    import sweetviz as sv  # lazy - see module docstring

    target = target_col if is_valid_sweetviz_target(df, target_col) else None
    report = sv.analyze(df, target_feat=target)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "sweetviz_report.html"
        report.show_html(
            filepath=str(out_path),
            open_browser=False,
            layout="vertical",
        )
        return out_path.read_text(encoding="utf-8")
