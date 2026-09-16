"""Adaptation policies for online non-stationary learning experiments.

Policies decide *whether* a model update should occur.  Keeping this decision
separate from the classifier and detector makes it possible to compare the
2018 CSD-triggered rule with periodic, warning-triggered, or later need-based
strategies under an otherwise identical pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol


@dataclass(frozen=True)
class AdaptationContext:
    """Information available when an adaptation decision is made."""

    trial_index: int
    time: float | int
    stage1_warning: bool = False
    validation_record: Mapping[str, object] | None = None


class AdaptationPolicy(Protocol):
    """Protocol implemented by adaptation-trigger rules."""

    def should_update(self, context: AdaptationContext) -> bool:
        """Return True when the classifier should be updated."""
        ...


@dataclass(frozen=True)
class NeverUpdate:
    """Non-adaptive baseline equivalent to EEG-NAC."""

    def should_update(self, context: AdaptationContext) -> bool:
        return False


@dataclass(frozen=True)
class RetrainOnWarning:
    """Retrain on every Stage-I warning, without Stage-II gating."""

    def should_update(self, context: AdaptationContext) -> bool:
        return bool(context.stage1_warning)


@dataclass(frozen=True)
class RetrainOnValidatedShift:
    """2018-style policy: retrain only after a confirmed Stage-II shift."""

    confirmed_field: str = "confirmed_shift"

    def should_update(self, context: AdaptationContext) -> bool:
        record = context.validation_record
        if record is None:
            return False
        return bool(record.get(self.confirmed_field, False))


@dataclass(frozen=True)
class PeriodicRetrain:
    """Passive baseline that retrains every ``interval`` evaluation trials."""

    interval: int = 10

    def __post_init__(self) -> None:
        if self.interval <= 0:
            raise ValueError("interval must be positive.")

    def should_update(self, context: AdaptationContext) -> bool:
        return (context.trial_index + 1) % self.interval == 0
