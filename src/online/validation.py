"""Streaming Stage-II: apply existing paper-aligned Hotelling test only when ready."""
from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy.stats import f

from src.detection.stage2.paper_two_sample_hotelling import (
    PaperTwoSampleHotellingConfig,
    calculate_paper_two_sample_hotelling,
)


@dataclass(frozen=True)
class PendingWarning:
    index: int
    time: int


class OnlineHotellingValidator:
    """Finite memory two-window validator. No post-warning lookahead."""

    def __init__(self, config: PaperTwoSampleHotellingConfig) -> None:
        self.config = config
        self.history: deque[tuple[int, int, np.ndarray]] = deque(
            maxlen=2 * config.window_size + 1
        )
        self.pending: list[PendingWarning] = []
        self.index = -1
        self.last_evaluated_index: int | None = None

    def update_one(
        self, time: int, projected: np.ndarray, warning: bool
    ) -> list[dict[str, object]]:
        sample = np.asarray(projected, dtype=float)
        if sample.ndim != 1 or sample.size < 1 or not np.isfinite(sample).all():
            raise ValueError("projected features must be a finite 1D vector")
        if self.history and time <= self.history[-1][1]:
            raise ValueError("time must increase")
        if self.history and sample.shape != self.history[-1][2].shape:
            raise ValueError("projected feature dimension must remain fixed")
        self.index += 1
        index = self.index
        self.history.append((index, int(time), sample.copy()))
        records: list[dict[str, object]] = []
        h = self.config.window_size
        gap = h if self.config.minimum_alarm_gap is None else self.config.minimum_alarm_gap

        if warning:
            if self.last_evaluated_index is not None and index - self.last_evaluated_index < gap:
                records.append(dict(alarm_time=int(time), validation_time=None,
                                    status="skipped_nearby_alarm", confirmed_shift=False))
            elif index < h - 1:
                records.append(dict(alarm_time=int(time), validation_time=None,
                                    status="insufficient_reference_window", confirmed_shift=False))
            else:
                self.pending.append(PendingWarning(index=index, time=int(time)))
                self.last_evaluated_index = index
                records.append(dict(alarm_time=int(time), validation_time=None,
                                    status="pending_current_window", confirmed_shift=False))

        still_pending = []
        for pending in self.pending:
            if index < pending.index + h:
                still_pending.append(pending)
                continue
            before = [row for row in self.history if pending.index-h+1 <= row[0] <= pending.index]
            after = [row for row in self.history if pending.index < row[0] <= pending.index+h]
            if len(before) != h or len(after) != h:
                raise RuntimeError("Stage-II history buffer lost required observations")
            record: dict[str, object] = dict(
                alarm_time=pending.time,
                validation_time=int(after[-1][1]),
                reference_start_time=int(before[0][1]),
                reference_end_time=int(before[-1][1]),
                current_start_time=int(after[0][1]),
                current_end_time=int(after[-1][1]),
                sample_size=h,
                n_features=int(sample.size),
                confirmed_shift=False,
            )
            try:
                t2, f_stat, df1, df2, p_value = calculate_paper_two_sample_hotelling(
                    current_sample=np.stack([row[2] for row in after]),
                    reference_sample=np.stack([row[2] for row in before]),
                    covariance_method=self.config.covariance_method,
                    regularization=self.config.regularization,
                )
            except ValueError as exc:
                if "too small for the number of features" not in str(exc):
                    raise
                record["status"] = "insufficient_degrees_of_freedom"
            else:
                record.update(
                    hotelling_t_squared=float(t2),
                    f_statistic=float(f_stat),
                    degrees_of_freedom_1=int(df1),
                    degrees_of_freedom_2=int(df2),
                    critical_value=float(f.ppf(1.0-self.config.alpha, df1, df2)),
                    p_value=float(p_value),
                    status="confirmed" if p_value < self.config.alpha else "rejected",
                    confirmed_shift=bool(p_value < self.config.alpha),
                )
            records.append(record)
        self.pending = still_pending
        return records

    def finish(self) -> list[dict[str, object]]:
        """Finalise unresolved warnings without inventing missing future data."""
        events = [
            dict(alarm_time=p.time, validation_time=None,
                 status="pending_current_window", confirmed_shift=False)
            for p in self.pending
        ]
        self.pending.clear()
        return events
