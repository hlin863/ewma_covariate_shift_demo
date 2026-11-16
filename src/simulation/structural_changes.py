"""Controlled, two-channel piecewise AR(1) structural-change simulations.

The scenarios isolate changes to stationary-target mean, zero-lag covariance
variance/correlation, or autoregressive dependence. This is a Park-inspired
illustration, NOT a reproduction of Park et al. (2023) wavelet/DPCA tests.

The scalar-AR model is X[t] = mu + phi * (X[t-1] - mu) + eps[t].
Stationary covariance Sigma and innovation covariance Q obey
Q = (1 - phi**2) * Sigma. Therefore ar_shift can hold Sigma constant.
At the change point, sample moments converge toward post-change stationary
targets; transient moments need not equal those targets immediately.
"""
import numpy as np


def _regime(mean, variance, correlation, ar):
    """Construct coherent stationary and innovation covariance matrices."""
    stationary_covariance = float(variance) * np.array(
        [[1.0, correlation], [correlation, 1.0]], dtype=float
    )
    return {
        "mean": np.asarray(mean, dtype=float).copy(),
        "phi": float(ar),
        "stationary_covariance": stationary_covariance,
        "innovation_covariance": (1.0 - ar**2) * stationary_covariance,
    }


def generate_structural_stream(
    seed: int,
    scenario: str,
    *,
    n_observations: int = 1000,
    change_at: int = 500,
    variance_ratio: float = 4.0,
    correlation_before: float = 0.0,
    correlation_after: float = 0.7,
    ar_before: float = 0.0,
    ar_after: float = 0.7,
) -> tuple[np.ndarray, dict]:
    """Generate an isolated structural-change stream and effective metadata.

    `variance_ratio` scales *stationary marginal variance*, not innovation
    variance. Names `covariance_shift` and `autocorrelation_shift` remain
    supported as aliases for correlation_shift and ar_shift respectively.
    `default` is a historical alias for combined, retained for notebook
    compatibility; new experiments should name their scenario explicitly.
    """
    if not isinstance(n_observations, (int, np.integer)) or n_observations < 2:
        raise ValueError("n_observations must be an integer >= 2.")
    if not isinstance(change_at, (int, np.integer)) or not 0 <= change_at <= n_observations:
        raise ValueError("change_at must be in [0, n_observations].")
    if not np.isfinite(variance_ratio) or variance_ratio <= 0:
        raise ValueError("variance_ratio must be finite and positive.")
    for name, value in (
        ("correlation_before", correlation_before),
        ("correlation_after", correlation_after),
        ("ar_before", ar_before),
        ("ar_after", ar_after),
    ):
        if not np.isfinite(value) or abs(value) >= 1:
            raise ValueError(f"{name} must be finite and strictly between -1 and 1.")

    aliases = {
        "covariance_shift": "correlation_shift",
        "autocorrelation_shift": "ar_shift",
        "default": "combined",
    }
    scenario_key = str(scenario).lower()
    resolved = aliases.get(scenario_key, scenario_key)
    valid = {"stationary", "mean_shift", "variance_shift", "correlation_shift", "ar_shift", "combined"}
    if resolved not in valid:
        raise ValueError(f"Unknown scenario: {scenario!r}. Choose from {sorted(valid)}.")

    # Target parameters before and after the change. Only explicitly selected
    # fields change. For the scalar AR coefficient, hold stationary Sigma fixed
    # in ar_shift by adjusting innovation covariance Q accordingly.
    baseline = {
        "mean": np.zeros(2, dtype=float),
        "variance": 1.0,
        "correlation": float(correlation_before),
        "ar": float(ar_before),
    }
    after_params = {
        "mean": baseline["mean"].copy(),
        "variance": baseline["variance"],
        "correlation": baseline["correlation"],
        "ar": baseline["ar"],
    }
    if resolved in ("mean_shift", "combined"):
        after_params["mean"] = np.array([2.0, 1.0], dtype=float)
    if resolved in ("variance_shift", "combined"):
        after_params["variance"] = float(variance_ratio)
    if resolved in ("correlation_shift", "combined"):
        after_params["correlation"] = float(correlation_after)
    if resolved in ("ar_shift", "combined"):
        after_params["ar"] = float(ar_after)

    before = _regime(**baseline)
    after = _regime(**after_params)
    rng = np.random.default_rng(seed)
    stream = np.empty((n_observations, 2), dtype=float)

    # At change_at=0 the first sample is post-change; at n, no samples
    # come from the post-change regime. Draw X0 from the selected target
    # stationary covariance to avoid an artificial initialization bias.
    initial = before if 0 < change_at else after
    stream[0] = rng.multivariate_normal(
        initial["mean"], initial["stationary_covariance"]
    )
    for t in range(1, n_observations):
        regime = before if t < change_at else after
        innovation = rng.multivariate_normal(
            np.zeros(2, dtype=float), regime["innovation_covariance"]
        )
        stream[t] = (
            regime["mean"]
            + regime["phi"] * (stream[t - 1] - regime["mean"])
            + innovation
        )

    def serialise(regime):
        return {
            "mean": regime["mean"].tolist(),
            "ar": regime["phi"],
            "stationary_covariance": regime["stationary_covariance"].tolist(),
            "innovation_covariance": regime["innovation_covariance"].tolist(),
        }

    metadata = {
        "scenario": scenario,
        "resolved_scenario": resolved,
        "model": "two-feature scalar AR(1); stationary covariance parameterization",
        "n_observations": int(n_observations),
        "change_at": int(change_at),
        "variance_ratio": float(variance_ratio),
        "correlation_before": float(correlation_before),
        "correlation_after": float(correlation_after),
        "ar_before": float(ar_before),
        "ar_after": float(ar_after),
        "before": serialise(before),
        "after": serialise(after),
        "shape": stream.shape,
    }
    return stream, metadata
