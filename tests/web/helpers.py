"""Small shared artifact writers for web tests."""

from pathlib import Path

import pandas as pd


COLUMNS = [
    "dataset",
    "subject",
    "lambda",
    "published_csw",
    "computed_csw",
    "csw_difference",
    "published_csv",
    "computed_csv",
    "csv_difference",
]


def write_table1_results(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            ["2A", "A01", 0.50, 12, 16, 4, 6, 1, -5],
            ["2A", "A02", 0.55, 15, 16, 1, 8, 2, -6],
            ["2B", "B01", 0.28, 14, 17, 3, 10, 0, -10],
            ["2B", "B02", 0.17, 18, 13, -5, 13, 1, -12],
        ],
        columns=COLUMNS,
    ).to_csv(path, index=False)
