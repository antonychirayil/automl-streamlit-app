"""
Shared test setup.

conftest.py is a file pytest imports automatically before running any
test, so anything defined here is available to every test file without
being imported. Two jobs here: put the project directory on the import
path (so `import preprocessing` works when pytest is run from the
project root), and define the small synthetic datasets the tests share.

A @pytest.fixture is just a named piece of setup: any test that takes an
argument called `classification_df` gets a freshly built one, so no test
can pollute another by modifying it.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# tests/ lives inside the project, so the project root is one level up.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def classification_df() -> pd.DataFrame:
    """
    A small, deliberately learnable classification dataset: `target` is
    mostly driven by `score`, so a model should beat guessing, plus a
    categorical column and one with missing values to exercise the
    imputation and encoding paths.
    """
    rng = np.random.default_rng(0)
    n = 200
    score = rng.normal(50, 10, n)
    return pd.DataFrame({
        "score": score,
        "age": rng.integers(20, 60, n),
        "region": rng.choice(["north", "south", "east"], n),
        "with_gaps": np.where(rng.random(n) < 0.2, np.nan, rng.normal(0, 1, n)),
        "target": (score > 50).astype(int),
    })


@pytest.fixture
def regression_df() -> pd.DataFrame:
    """Continuous target with a genuine linear signal plus noise."""
    rng = np.random.default_rng(1)
    n = 200
    size = rng.normal(100, 20, n)
    rooms = rng.integers(1, 6, n)
    return pd.DataFrame({
        "size": size,
        "rooms": rooms,
        "district": rng.choice(["a", "b"], n),
        "price": size * 3.5 + rooms * 10 + rng.normal(0, 5, n),
    })
