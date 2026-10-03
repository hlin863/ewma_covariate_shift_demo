"""Shared event helpers for detector-driven adaptation loops."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _validation_by_time(
    validation_results: pd.DataFrame,
) -> dict[object, dict[str, object]]:
    if validation_results.empty:
        return {}

    if "validation_time" in validation_results.columns:
        time_column = "validation_time"
    elif "alarm_time" in validation_results.columns:
        time_column = "alarm_time"
    else:
        raise ValueError(
            "validation_results must contain 'validation_time' or 'alarm_time'."
        )

    records: dict[object, dict[str, object]] = {}
    for row in validation_results.to_dict(orient="records"):
        value = row.get(time_column)
        if value is None or (isinstance(value, float) and np.isnan(value)):
            continue
        records[value] = row
    return records


def _warning_times(warning_results: pd.DataFrame | None) -> set[object]:
    if warning_results is None or warning_results.empty:
        return set()
    required = {"time", "stage_1_alarm"}
    if not required.issubset(warning_results.columns):
        raise ValueError(
            "warning_results must contain 'time' and 'stage_1_alarm' columns."
        )
    alarms = warning_results.loc[warning_results["stage_1_alarm"].astype(bool), "time"]
    return set(alarms.tolist())


def _python_scalar(value: object) -> object:
    return value.item() if hasattr(value, "item") else value
