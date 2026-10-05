"""
AutoML Streamlit app - entry point.

Upload a CSV, pick the column you want to predict, and the app takes it
from there: it works out whether that's a classification or a regression
problem, preprocesses the data, trains and compares five models, tunes
their hyperparameters, charts the results, and will score a second,
unseen file with whichever model won.

This file is deliberately tiny. Its whole job is to set the page up and
call two functions, so that the shape of the app is obvious from one
screen. The work lives in these modules:

    config.py          every tunable number, with the reasoning
    state.py           session state + cached (expensive) work
    ui_sidebar.py      all input controls - WRITES to session state
    ui_main.py         all output rendering - READS session state
    preprocessing.py   splitting, imputing, encoding, scaling, selection
    models.py          model definitions, training, hyperparameter search
    evaluation.py      metrics and picking a winner
    visualization.py   matplotlib figures
    profiling.py       optional one-line EDA reports (ydata / sweetviz)

The four modules in the bottom group import no Streamlit at all: they
are plain Python that takes data and returns results, which is what
makes them testable by the suite in tests/ without a browser anywhere
in sight.

Run it with:  streamlit run app.py
"""

import streamlit as st

import state
from ui_main import render_main
from ui_sidebar import render_sidebar


def main() -> None:
    # Must be the first Streamlit call on the page.
    st.set_page_config(page_title="AutoML App", page_icon="🤖", layout="wide")

    # Streamlit re-runs this entire file on every interaction, so the
    # first job is always to make sure the remembered values exist.
    state.init_state()

    # Order matters: the sidebar runs first so that anything the user
    # just clicked has already been written to session state by the time
    # the main area reads it to render this pass.
    render_sidebar()
    render_main()


if __name__ == "__main__":
    main()
