"""Read-only multivariate time-series inspection inspired by Park et al. (2023).

This module is NOT a reproduction of the paper's aggregated CUSUM,
locally stationary wavelet spectral estimator, or dynamic PCA.
It prepares equally sampled multivariate streams and descriptive windows.
"""
from pathlib import Path

import numpy as np
import pandas as pd


def load_multivariate_csv(path: str | Path) -> pd.DataFrame:
    """Load a time-ordered CSV with >=2 numeric sensor columns.

    An optional 'time' column must be numeric and strictly increasing.
    Missing/non-finite observations are rejected rather than interpolated,
    so the analysis never silently changes the temporal structure.
    """
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError("Multivariate CSV must contain observations.")
    if "time" in frame.columns:
        times = pd.to_numeric(frame["time"], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
            raise ValueError("time must be finite and strictly increasing.")
        frame["time"] = times
    else:
        frame.insert(0, "time", np.arange(len(frame), dtype=float))
    channels = [name for name in frame.columns if name != "time"]
    if len(channels) < 2:
        raise ValueError("At least two sensor columns are required.")
    for name in channels:
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    if not np.isfinite(frame[channels].to_numpy(dtype=float)).all():
        raise ValueError("Sensor observations must be finite numeric values.")
    if len(frame) < 8:
        raise ValueError("At least eight observations are required.")
    return frame


def demonstration_stream(seed: int = 42, size: int = 240) -> pd.DataFrame:
    """Deterministic synthetic demonstration, NOT a paper simulation."""
    if size < 24 or size % 2:
        raise ValueError("Demo size must be even and at least 24.")
    rng = np.random.default_rng(seed)
    first = rng.standard_normal((size // 2, 3))
    second = rng.standard_normal((size // 2, 3))
    # Similar marginal mean/variance, but changed cross-channel dependence.
    first[:, 1] = 0.9 * first[:, 0] + np.sqrt(1 - 0.9**2) * first[:, 1]
    second[:, 1] = -0.9 * second[:, 0] + np.sqrt(1 - 0.9**2) * second[:, 1]
    return pd.DataFrame(np.vstack((first, second)), columns=["sensor_1", "sensor_2", "sensor_3"]).assign(
        time=np.arange(size, dtype=float)
    )[["time", "sensor_1", "sensor_2", "sensor_3"]]


def describe_windows(frame: pd.DataFrame, window_size: int = 40) -> list[dict]:
    """Non-overlapping window mean, variance and correlation descriptors.

    Window metrics are descriptive (not significance tests or alarms).
    Incomplete trailing observations are explicitly excluded.
    """
    if window_size < 4:
        raise ValueError("window_size must be at least four.")
    columns = [c for c in frame.columns if c != "time"]
    if len(columns) < 2:
        raise ValueError("At least two sensor columns are required.")
    output = []
    for start in range(0, len(frame) - window_size + 1, window_size):
        block = frame.iloc[start:start + window_size]
        values = block[columns].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Non-finite sensor observations are not supported.")
        corr = np.corrcoef(values, rowvar=False)
        output.append({
            "window": len(output) + 1,
            "start": round(float(block["time"].iloc[0]), 5),
            "end": round(float(block["time"].iloc[-1]), 5),
            "mean": round(float(np.mean(values)), 4),
            "variance": round(float(np.mean(np.var(values, axis=0, ddof=1))), 4),
            "corr_12": round(float(corr[0, 1]), 4) if np.isfinite(corr[0, 1]) else None,
        })
    return output
