"""Experiment result formatting and evaluation helpers."""

from src.reporting.metrics import (
    DetectionMetrics,
    RepeatedShiftEvaluation,
    RepeatedShiftMetrics,
    evaluate_repeated_shift_detection,
    evaluate_stage_1_detection,
)
from src.reporting.table1 import (
    Table1Row,
    comparison_dataframe,
    paper_style_dataframe,
    paper_style_markdown,
)

__all__ = [
    "DetectionMetrics",
    "RepeatedShiftEvaluation",
    "RepeatedShiftMetrics",
    "Table1Row",
    "comparison_dataframe",
    "evaluate_repeated_shift_detection",
    "evaluate_stage_1_detection",
    "paper_style_dataframe",
    "paper_style_markdown",
]
