"""Adaptive H around Stage-I warnings; reuses existing Hotelling statistic."""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import pandas as pd

from src.detection.stage2.paper_two_sample_hotelling import (
    calculate_paper_two_sample_hotelling,
)
from src.detection.stage2.window_feasibility import (
    WindowSelectionPolicy,
    select_validation_window,
)


@dataclass(frozen=True)
class AdaptiveHotellingConfig:
    window_policy: WindowSelectionPolicy = field(default_factory=WindowSelectionPolicy)
    alpha: float = 0.05
    # Fixed, protocol-controlled warning spacing; never silently tie it to selected H.
    minimum_alarm_gap: int = 10

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        if self.minimum_alarm_gap < 0:
            raise ValueError("minimum_alarm_gap must be >= 0")


def validate_adaptive_window_alarms(
    features: np.ndarray,
    times: np.ndarray,
    stage_1_results: pd.DataFrame,
    config: AdaptiveHotellingConfig | None = None,
) -> pd.DataFrame:
    """Select H from pre-warning history, then validate after H new trials.

    Full-stream input is processed as though future observations arrive later.
    This is retrospective evaluation; no post-warning samples influence H.
    """
    cfg = config or AdaptiveHotellingConfig()
    observations = np.asarray(features, dtype=float)
    time_values = np.asarray(times)
    if observations.ndim != 2 or min(observations.shape) == 0:
        raise ValueError("features must have shape (n, d) with n,d > 0")
    if not np.isfinite(observations).all():
        raise ValueError("features must be finite")
    if time_values.ndim != 1 or len(time_values) != len(observations):
        raise ValueError("times must have one value per observation")
    if np.unique(time_values).size != time_values.size:
        raise ValueError("times must be unique")
    if not {"time", "stage_1_alarm"}.issubset(stage_1_results.columns):
        raise ValueError("stage_1_results requires time and stage_1_alarm")

    columns = [
        "alarm_time",
        "validation_time",
        "reference_start_time",
        "reference_end_time",
        "current_start_time",
        "current_end_time",
        "sample_size",
        "n_features",
        "required_min_window",
        "reference_condition_number",
        "hotelling_t_squared",
        "f_statistic",
        "degrees_of_freedom_1",
        "degrees_of_freedom_2",
        "p_value",
        "status",
        "confirmed_shift",
    ]
    alarm_times = (
        stage_1_results.loc[stage_1_results["stage_1_alarm"].astype(bool), "time"]
        .drop_duplicates()
        .to_numpy()
    )
    location = {t: i for i, t in enumerate(time_values.tolist())}
    try:
        alarm_indices = sorted(int(location[t]) for t in alarm_times)
    except KeyError as exc:
        raise ValueError("Stage-I alarm time not present in feature timeline") from exc

    records: list[dict[str, object]] = []
    last_evaluated_index: int | None = None
    for i in alarm_indices:
        record: dict[str, object] = dict.fromkeys(columns, None)
        record.update(
            {
                "alarm_time": time_values[i],
                "n_features": int(observations.shape[1]),
                "degrees_of_freedom_1": int(observations.shape[1]),
                "confirmed_shift": False,
            }
        )
        if (
            last_evaluated_index is not None
            and i - last_evaluated_index < cfg.minimum_alarm_gap
        ):
            record["status"] = "skipped_nearby_alarm"
            records.append(record)
            continue

        choice = select_validation_window(observations[: i + 1], cfg.window_policy)
        record["required_min_window"] = choice.required_min_window
        record["reference_condition_number"] = choice.condition_number
        record["status"] = choice.status
        if choice.status != "selected":
            records.append(record)
            continue

        h = int(choice.window_size)
        record["sample_size"] = h
        record["degrees_of_freedom_2"] = choice.df2
        record["reference_start_time"] = time_values[i - h + 1]
        record["reference_end_time"] = time_values[i]
        if i + 1 + h > len(observations):
            record["status"] = "pending_current_window"
            records.append(record)
            continue

        before = observations[i - h + 1 : i + 1]
        after = observations[i + 1 : i + 1 + h]
        record["current_start_time"] = time_values[i + 1]
        record["current_end_time"] = time_values[i + h]
        record["validation_time"] = time_values[i + h]
        t2, f_stat, df1, df2, p = calculate_paper_two_sample_hotelling(
            current_sample=after,
            reference_sample=before,
            covariance_method=cfg.window_policy.covariance_method,
            regularization=cfg.window_policy.regularization,
        )
        confirmed = bool(p < cfg.alpha)
        record.update(
            {
                "hotelling_t_squared": t2,
                "f_statistic": f_stat,
                "degrees_of_freedom_1": df1,
                "degrees_of_freedom_2": df2,
                "p_value": p,
                "status": "confirmed" if confirmed else "rejected",
                "confirmed_shift": confirmed,
            }
        )
        records.append(record)
        last_evaluated_index = i
    return pd.DataFrame.from_records(records, columns=columns)
