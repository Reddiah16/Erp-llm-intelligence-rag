"""Chart-ready views of the pilot metrics for the Streamlit dashboard.

Kept separate from dashboard/app.py so the transformations are importable and
runner-testable: during QA on 2026-11-07 a broken dict-comprehension in the
accuracy-by-route chart crashed the whole Evaluation page with NameError after
the page had already rendered, and the text-contract dashboard tests cannot see
a runtime fault like that. The chart frames these functions build are asserted
against real metric shapes instead.
"""
from typing import Any, Dict

import pandas as pd


def route_accuracy_frame(routes: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """One row ('accuracy %'), one column per route, a 0.0 for no data.

    A route with no graded turns reports value None; the chart needs a number,
    so the empty sample is drawn as 0.0 while the table beside it still says
    'Not enough data'.
    """
    return pd.DataFrame(
        {name: stats["value"] if stats["value"] is not None else 0.0
         for name, stats in routes.items()},
        index=["accuracy %"])
