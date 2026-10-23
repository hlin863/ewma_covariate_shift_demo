"""ABCD equation-level experiment: synthetic Gaussian streams, frozen PCA, Eq. (5)/(8).

Educational demonstration, NOT the authors' online ABCD Algorithm 2.
Sources: Heyden et al. (2024), and heymarco/StreamGenerators Gaussian.
- Two concepts instead of the paper's ten, and shorter browser-friendly streams.
- PCA is frozen, not reset after a detection.
- Sample variances are plugged into a population-variance Bernstein bound.
- Candidate scans are not calibrated as repeated sequential tests.
- Global MinMax normalization follows the reference preprocessing but leaks future data.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import numpy as np

from src.detection.baselines.abcd import bernstein_two_sample_upper_bound


@dataclass(frozen=True)
class ABCDLabConfig:
    scenario: str = "Normal-M"
    seed: int = 0
    segment_size: int = 200
    dimensions: int = 24
    changed_dimensions: int = 8
    bottleneck: int = 6
    shift: float = 1.5
    delta: float = 0.05
    M: float = 0.1
    candidate_count: int = 20

    def validate(self):
        if self.scenario not in {"Normal-M", "Normal-V"}:
            raise ValueError("Unknown scenario (use Normal-M or Normal-V).")
        if not isinstance(self.seed, int) or not 0 <= self.seed <= 2**32 - 1:
            raise ValueError("Seed must be a non-negative 32-bit integer.")
        if not 80 <= self.segment_size <= 500:
            raise ValueError("Segment size must be between 80 and 500.")
        if not 4 <= self.dimensions <= 64:
            raise ValueError("Dimensions must be between 4 and 64.")
        if not 1 <= self.changed_dimensions < self.dimensions:
            raise ValueError("Changed dimensions must be between 1 and d-1.")
        if not 1 <= self.bottleneck < self.dimensions:
            raise ValueError("Bottleneck must be between 1 and d-1.")
        if not math.isfinite(self.shift) or not 0.1 <= self.shift <= 3.0:
            raise ValueError("Shift/scale parameter must be in [0.1,3].")
        if not math.isfinite(self.delta) or not 0 < self.delta < 1:
            raise ValueError("Delta must be in (0,1).")
        if not math.isfinite(self.M) or self.M not in (0.1, 1.0):
            raise ValueError("M must be 0.1 (empirical) or 1.0 (theoretical).")
        if not 2 <= self.candidate_count <= 40:
            raise ValueError("Candidate count must be in [2,40].")


def _paper_inspired_gaussian(config: ABCDLabConfig):
    """Two-concept controlled subset inspired by the authors' Gaussian generator."""
    n, d, changed = config.segment_size, config.dimensions, config.changed_dimensions
    rng = np.random.default_rng(config.seed)
    before = rng.normal(size=(n, d))
    after = rng.normal(size=(n, d))
    if config.scenario == "Normal-M":
        after[:, :changed] += config.shift
    else:
        after[:, :changed] *= config.shift
    values = np.vstack((before, after))
    lo, hi = values.min(axis=0), values.max(axis=0)
    values = (values - lo) / np.where(hi > lo, hi - lo, 1.0)
    return values, n


def _pca_losses(values: np.ndarray, n_train: int, n_components: int):
    """Fit PCA on concept 1, freeze it, and reconstruct the entire stream."""
    training = values[:n_train]
    mu = training.mean(axis=0)
    _, _, vt = np.linalg.svd(training - mu, full_matrices=False)
    basis = vt[:n_components]
    projection = (values - mu) @ basis.T
    reconstruction = mu + projection @ basis
    reconstruction = np.clip(reconstruction, 0.0, 1.0)
    squared_errors = (values - reconstruction) ** 2
    losses = squared_errors.mean(axis=1)  # Heyden et al., Eq. (1)
    return losses, squared_errors, projection


def _split_row(losses: np.ndarray, k: int, M: float):
    left, right = losses[:k], losses[k:]
    mean1, mean2 = float(left.mean()), float(right.mean())
    v1, v2 = float(left.var(ddof=1)), float(right.var(ddof=1))
    eps = abs(mean1 - mean2)  # Eq. (8): observed mean difference
    kappa = float(np.clip(len(right) / len(losses), 0.05, 0.95))  # Eq. (12)
    e1, e2 = kappa * eps, (1 - kappa) * eps
    b1 = 2 * math.exp(-k * e1**2 / (2 * (v1 + M * e1 / 3))) if eps else 2.0
    b2 = 2 * math.exp(-(len(losses) - k) * e2**2 /
                      (2 * (v2 + M * e2 / 3))) if eps else 2.0
    score = bernstein_two_sample_upper_bound(
        k, len(losses) - k, v1, v2, eps, M, kappa
    )  # Eq. (5); sample-variance plug-in, not a calibrated p-value
    return {
        "k": int(k), "n1": int(k), "n2": int(len(losses) - k),
        "mean_before": mean1, "mean_after": mean2,
        "variance_before": v1, "variance_after": v2,
        "epsilon": eps, "kappa": kappa,
        "term_before": b1, "term_after": b2, "bound": score,
    }


def run_abcd_lab(config: ABCDLabConfig) -> dict:
    """Compute serialisable evidence; not benchmark precision/recall/F1."""
    config.validate()
    values, true_change = _paper_inspired_gaussian(config)
    losses, squared_errors, projection = _pca_losses(
        values, true_change, config.bottleneck
    )
    t = len(losses)
    positions = np.linspace(30, t - 30, num=config.candidate_count, dtype=int)
    candidates = sorted(set(int(k) for k in positions))
    rows = [_split_row(losses, k, config.M) for k in candidates]
    best = min(rows, key=lambda row: row["bound"])  # Eq. (8)
    return {
        "metadata": {
            "source": "Paper-inspired synthetic Normal-M/V; NOT a reproduced Heyden benchmark",
            "config": asdict(config),
            "true_change_after": true_change,
            "training_count": true_change, "monitor_count": t,
            "dataset_name": config.scenario, "representation": "frozen PCA",
            "normalisation": "global MinMax (future-data leakage for online evaluation)",
            "statistical_limit": "sample-variance plug-in; scan over splits; not a calibrated stream-wide p-value",
            "method_status": "batch teaching baseline, not full ABCD Algorithm 2",
        },
        "observations": [
            {"index": i + 1, "loss": float(losses[i]),
             "score_pc1": float(projection[i, 0])}
            for i in range(t)
        ],
        "feature_residuals": [float(x) for x in squared_errors.mean(axis=0)],
        "splits": rows,
        "best": best,
        "summary": {
            "actual_change_after": true_change,
            "estimated_split_after": int(best["k"]),
            "detection_offset": int(best["k"] - true_change),
            "min_raw_bound": float(best["bound"]),
            "nominal_flag": bool(best["bound"] < config.delta),
            "num_splits": len(rows),
            "pre_change_loss": float(losses[:true_change].mean()),
            "post_change_loss": float(losses[true_change:].mean()),
        },
    }
