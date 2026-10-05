"""
Tests for profiling.py - the optional EDA extras.

These deliberately do NOT require ydata-profiling or sweetviz to be
installed: the whole design goal of that module is that the app behaves
sensibly either way, and this suite has to be able to prove it in both
states.
"""

import numpy as np
import pandas as pd

import profiling


# --- availability checks ----------------------------------------------
def test_is_installed_answers_without_raising():
    """Whatever the answer, it must be a bool and must not blow up."""
    for info in profiling.PACKAGES.values():
        assert isinstance(profiling.is_installed(info["import_name"]), bool)


def test_a_definitely_absent_module_reports_false():
    assert profiling.is_installed("definitely_not_a_real_module_xyz") is False


def test_install_hint_names_the_pip_package():
    assert profiling.install_hint("Sweetviz") == "pip install sweetviz"
    assert "ydata-profiling" in profiling.install_hint("ydata-profiling")


# --- sampling ---------------------------------------------------------
def test_large_frames_are_sampled_down():
    df = pd.DataFrame({"x": range(100)})
    sample, was_sampled = profiling.maybe_sample(df, 30)

    assert len(sample) == 30
    assert was_sampled is True


def test_small_frames_are_left_alone():
    df = pd.DataFrame({"x": range(10)})
    sample, was_sampled = profiling.maybe_sample(df, 500)

    assert len(sample) == 10
    assert was_sampled is False


def test_sampling_is_reproducible():
    df = pd.DataFrame({"x": range(1000)})
    first, _ = profiling.maybe_sample(df, 50)
    second, _ = profiling.maybe_sample(df, 50)

    assert list(first["x"]) == list(second["x"])


# --- Sweetviz target eligibility --------------------------------------
def test_numeric_target_is_accepted():
    df = pd.DataFrame({"target": [0, 1, 0, 1]})
    assert profiling.is_valid_sweetviz_target(df, "target") is True


def test_text_target_is_rejected():
    """Sweetviz only supports numeric or boolean targets."""
    df = pd.DataFrame({"target": ["yes", "no"]})
    assert profiling.is_valid_sweetviz_target(df, "target") is False


def test_target_with_missing_values_is_rejected():
    df = pd.DataFrame({"target": [1.0, np.nan, 0.0]})
    assert profiling.is_valid_sweetviz_target(df, "target") is False


def test_absent_or_unset_target_is_rejected():
    df = pd.DataFrame({"target": [0, 1]})
    assert profiling.is_valid_sweetviz_target(df, None) is False
    assert profiling.is_valid_sweetviz_target(df, "no_such_column") is False


# --- error translation ------------------------------------------------
def test_pkg_resources_error_points_at_the_version_pin():
    """
    The bug this function exists for: `pip install setuptools` installs
    82+, which no longer contains pkg_resources, so the hint has to name
    the version cap rather than just the package.
    """
    error = ModuleNotFoundError("No module named 'pkg_resources'", name="pkg_resources")
    hint = profiling.explain_import_failure(error)

    assert "setuptools<82" in hint


def test_unmapped_module_falls_back_to_its_own_name():
    error = ModuleNotFoundError("No module named 'wordcloud'", name="wordcloud")
    assert "pip install wordcloud" in profiling.explain_import_failure(error)


def test_non_import_errors_are_not_translated():
    """Ordinary failures must keep their own message, not get a pip hint."""
    assert profiling.explain_import_failure(ValueError("something else")) is None
