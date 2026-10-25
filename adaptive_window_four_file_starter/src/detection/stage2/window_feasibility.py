"""Feasibility-aware selection of Stage-II equal validation windows.

The policy sees only the features available at Stage-I warning time.
This module never examines the future/current validation window.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
from sklearn.covariance import LedoitWolf


@dataclass(frozen=True)
class WindowSelectionPolicy:
    min_window: int = 2
    max_window: int = 40
    target_df2: int = 10
    covariance_method: str = "shrinkage"
    regularization: float = 1e-6
    max_condition_number: float = 1e6
    require_full_rank_empirical: bool = False

    def __post_init__(self) -> None:
        if self.min_window < 2 or self.max_window < self.min_window:
            raise ValueError("Require 2 <= min_window <= max_window")
        if self.target_df2 < 1:
            raise ValueError("target_df2 must be >= 1")
        if self.covariance_method not in {"shrinkage", "empirical"}:
            raise ValueError("Unsupported covariance_method")
        if self.regularization < 0:
            raise ValueError("regularization must be >= 0")
        if self.max_condition_number <= 1:
            raise ValueError("max_condition_number must be > 1")


@dataclass(frozen=True)
class WindowSelection:
    status: str
    required_min_window: int
    window_size: int | None = None
    df2: int | None = None
    condition_number: float | None = None


def minimum_equal_window_size(d: int, target_df2: int = 1) -> int:
    """Smallest integer H satisfying 2H - d - 1 >= target_df2."""
    if d < 1 or target_df2 < 1:
        raise ValueError("d and target_df2 must be positive")
    return max(2, math.ceil((d + 1 + target_df2) / 2))


def _condition_number(reference: np.ndarray, policy: WindowSelectionPolicy) -> float:
    d = reference.shape[1]
    if policy.covariance_method == "shrinkage":
        covariance = LedoitWolf().fit(reference).covariance_
    else:
        covariance = np.atleast_2d(np.cov(reference, rowvar=False, ddof=1))
    regularized = covariance + policy.regularization * np.eye(d)
    return float(np.linalg.cond(regularized))


def select_validation_window(
    history_through_alarm: np.ndarray,
    policy: WindowSelectionPolicy,
) -> WindowSelection:
    """Select smallest feasible H using ONLY history through current warning.

    history_through_alarm includes the observation at the warning index.
    Failure statuses are intentional data, not numerical confirmation failures.
    """
    history = np.asarray(history_through_alarm, dtype=float)
    if history.ndim != 2 or history.shape[1] == 0:
        raise ValueError("history must have shape (n, d), d >= 1")
    if not np.isfinite(history).all():
        raise ValueError("history must be finite")

    n_history, d = history.shape
    required = max(
        policy.min_window,
        minimum_equal_window_size(d, policy.target_df2),
        d + 1 if policy.require_full_rank_empirical else 2,
    )
    if required > policy.max_window:
        return WindowSelection("infeasible_policy", required)
    if n_history < required:
        return WindowSelection("insufficient_reference_window", required)

    for h in range(required, min(n_history, policy.max_window) + 1):
        reference = history[-h:]
        if policy.require_full_rank_empirical:
            empirical = np.atleast_2d(np.cov(reference, rowvar=False, ddof=1))
            if np.linalg.matrix_rank(empirical) < d:
                continue
        condition = _condition_number(reference, policy)
        if np.isfinite(condition) and condition <= policy.max_condition_number:
            return WindowSelection(
                status="selected",
                required_min_window=required,
                window_size=h,
                df2=2 * h - d - 1,
                condition_number=condition,
            )
    return WindowSelection("unstable_reference_covariance", required)
