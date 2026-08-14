import numpy as np

from src.table1_reproduction import (
    Table1Row,
    comparison_dataframe,
    paper_style_dataframe,
    paper_style_markdown,
)


def _rows() -> list[Table1Row]:
    rows = []
    for index in range(1, 10):
        rows.append(Table1Row("2A", f"A{index:02d}", 0.50, 12, 6, 10 + index, 4 + index))
        rows.append(Table1Row("2B", f"B{index:02d}", 0.28, 14, 10, 12 + index, 8 + index))
    return rows


def test_paper_style_dataframe_has_nine_subject_rows_and_mean() -> None:
    frame = paper_style_dataframe(_rows())
    assert frame.shape == (10, 8)
    assert frame.iloc[0]["2A Subject"] == "A01"
    assert frame.iloc[0]["2B Subject"] == "B01"
    assert frame.iloc[-1]["2A Subject"] == "Mean"
    assert frame.iloc[-1]["2B Subject"] == "Mean"
    assert np.isclose(frame.iloc[-1]["2A λ"], 0.50)
    assert np.isclose(frame.iloc[-1]["2B λ"], 0.28)


def test_comparison_dataframe_retains_published_and_computed_values() -> None:
    frame = comparison_dataframe(_rows())
    assert frame.shape[0] == 18
    assert {"published_csw", "computed_csw", "csw_difference"}.issubset(frame.columns)
    assert {"published_csv", "computed_csv", "csv_difference"}.issubset(frame.columns)


def test_markdown_uses_grouped_2a_2b_structure() -> None:
    text = paper_style_markdown(_rows())
    assert "2A Subject" in text
    assert "2B Subject" in text
    assert "A01" in text
    assert "B09" in text
    assert "Mean" in text
