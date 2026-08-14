"""Formatting helpers for the CSE Table 1 reproduction experiment."""

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Table1Row:
    dataset: str
    subject: str
    lambda_value: float
    published_csw: int
    published_csv: int
    computed_csw: int
    computed_csv: int

    def long_row(self) -> dict[str, object]:
        return {
            "dataset": self.dataset,
            "subject": self.subject,
            "lambda": self.lambda_value,
            "published_csw": self.published_csw,
            "computed_csw": self.computed_csw,
            "csw_difference": self.computed_csw - self.published_csw,
            "published_csv": self.published_csv,
            "computed_csv": self.computed_csv,
            "csv_difference": self.computed_csv - self.published_csv,
        }


def _mean_row(rows: list[Table1Row]) -> tuple[str, float, float, float]:
    if not rows:
        raise ValueError("at least one result row is required.")
    return (
        "Mean",
        sum(row.lambda_value for row in rows) / len(rows),
        sum(row.computed_csw for row in rows) / len(rows),
        sum(row.computed_csv for row in rows) / len(rows),
    )


def paper_style_dataframe(rows: list[Table1Row]) -> pd.DataFrame:
    """Return computed 2A and 2B results in the side-by-side paper layout."""

    rows_2a = sorted(
        [row for row in rows if row.dataset == "2A"], key=lambda row: row.subject
    )
    rows_2b = sorted(
        [row for row in rows if row.dataset == "2B"], key=lambda row: row.subject
    )
    if len(rows_2a) != len(rows_2b):
        raise ValueError("paper-style output requires equal numbers of 2A and 2B rows.")

    output: list[dict[str, object]] = []
    for left, right in zip(rows_2a, rows_2b):
        output.append({
            "2A Subject": left.subject,
            "2A λ": left.lambda_value,
            "2A CSW": left.computed_csw,
            "2A CSV": left.computed_csv,
            "2B Subject": right.subject,
            "2B λ": right.lambda_value,
            "2B CSW": right.computed_csw,
            "2B CSV": right.computed_csv,
        })

    mean_2a = _mean_row(rows_2a)
    mean_2b = _mean_row(rows_2b)
    output.append({
        "2A Subject": mean_2a[0],
        "2A λ": mean_2a[1],
        "2A CSW": mean_2a[2],
        "2A CSV": mean_2a[3],
        "2B Subject": mean_2b[0],
        "2B λ": mean_2b[1],
        "2B CSW": mean_2b[2],
        "2B CSV": mean_2b[3],
    })
    return pd.DataFrame(output)


def comparison_dataframe(rows: list[Table1Row]) -> pd.DataFrame:
    """Return long-form published-versus-computed reproduction diagnostics."""

    return pd.DataFrame([row.long_row() for row in rows])


def _format_value(value: object, *, decimal: bool = False) -> str:
    if isinstance(value, str):
        return value
    if decimal:
        return f"{float(value):.2f}"
    if isinstance(value, float) and not value.is_integer():
        return f"{value:.2f}"
    return str(int(value))


def paper_style_markdown(rows: list[Table1Row]) -> str:
    """Render computed results in the paper's grouped 2A/2B table structure."""

    frame = paper_style_dataframe(rows)
    headers = list(frame.columns)
    lines = [
        "# Table 1 reproduction: CSE on BCI Competition IV Dataset 2A and 2B",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    decimal_columns = {"2A λ", "2B λ", "2A CSW", "2A CSV", "2B CSW", "2B CSV"}
    for _, row in frame.iterrows():
        cells = [
            _format_value(row[column], decimal=column in decimal_columns)
            for column in headers
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)
