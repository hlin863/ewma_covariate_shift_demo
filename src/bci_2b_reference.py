"""Published CSE reference values for BCI Competition IV Dataset 2B.

These constants are transcribed from Table 1 supplied for the replication
work. They are reference targets only; they must never be substituted for
experiment output.
"""

from __future__ import annotations

import pandas as pd


BCI_2B_TABLE1_ROWS: tuple[dict[str, float | int | str], ...] = (
    {"subject": "B01", "paper_lambda": 0.28, "paper_csw": 14, "paper_csv": 10},
    {"subject": "B02", "paper_lambda": 0.17, "paper_csw": 18, "paper_csv": 13},
    {"subject": "B03", "paper_lambda": 0.60, "paper_csw": 19, "paper_csv": 12},
    {"subject": "B04", "paper_lambda": 0.20, "paper_csw": 11, "paper_csv": 6},
    {"subject": "B05", "paper_lambda": 0.10, "paper_csw": 12, "paper_csv": 8},
    {"subject": "B06", "paper_lambda": 0.33, "paper_csw": 22, "paper_csv": 12},
    {"subject": "B07", "paper_lambda": 0.30, "paper_csw": 17, "paper_csv": 11},
    {"subject": "B08", "paper_lambda": 0.21, "paper_csw": 27, "paper_csv": 14},
    {"subject": "B09", "paper_lambda": 0.45, "paper_csw": 18, "paper_csv": 7},
)


def bci_2b_table1_reference() -> pd.DataFrame:
    """Return a copy of the published Dataset 2B Table 1 targets."""

    return pd.DataFrame(BCI_2B_TABLE1_ROWS).copy()
