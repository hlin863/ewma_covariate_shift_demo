"""Configuration for operational TWIN-ADAPT."""

from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class TwinAdaptConfig:
    # Observation delivery
    batch_size: int = 40

    # Accuracy monitoring
    win_size: int = 10

    # Maximum adaptive buffer size, in trials
    win_size_max: int = 120

    # Relative accuracy thresholds
    gamma: float = 0.95
    delta: float = 0.85

    # Label delay measured in complete batches
    # None means no labels released to learner
    label_delay: int | None = 0

    # Reproducibility
    seed: int = 42

    # Explicit algorithm interpretation
    algorithm_mode: str = "operational"

    def __post_init__(self):
        for name in (
            "batch_size",
            "win_size",
            "win_size_max",
        ):
            value = getattr(self, name)

            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

        if not 0 < self.win_size < self.batch_size:
            raise ValueError("Require 0 < win_size < batch_size")

        if self.win_size_max < self.batch_size:
            raise ValueError("win_size_max must hold at least one batch")

        if not 0 < self.delta < self.gamma < 1:
            raise ValueError("Require 0 < delta < gamma < 1")

        if self.label_delay is not None:
            if type(self.label_delay) is not int or self.label_delay < 0:
                raise ValueError("label_delay must be nonnegative or None")

        if type(self.seed) is not int:
            raise ValueError("seed must be an integer")

        if not 0 <= self.seed < 2**32:
            raise ValueError("seed outside valid range")

        if self.algorithm_mode != "operational":
            raise ValueError("Unsupported algorithm interpretation")

    def snapshot(self) -> dict:
        return {
            **asdict(self),
            "max_window_unit": "trials",
            "label_delay_unit": "batches",
        }
