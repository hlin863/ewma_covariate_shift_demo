"""Generic feature extraction helpers for motor-imagery EEG experiments.

This module contains the repo-specific sliding-window band-power feature path.
It is intentionally separate from the paper-aligned FBCSP pipeline in
``src.bci.fbcsp``.
"""

from dataclasses import dataclass

import numpy as np
from scipy.signal import welch

from src.bci.data import BCISessionData


@dataclass(frozen=True)
class WindowedFeatureResult:
    features: np.ndarray
    times: np.ndarray
    feature_names: tuple[str, ...]
    session_ids: np.ndarray


def extract_bandpower_windows(
    session: BCISessionData,
    *,
    window_seconds: float = 2.0,
    step_seconds: float = 0.5,
    frequency_bands: tuple[tuple[str, float, float], ...] = (
        ("mu", 8.0, 13.0),
        ("beta", 13.0, 30.0),
    ),
) -> WindowedFeatureResult:
    if window_seconds <= 0.0 or step_seconds <= 0.0:
        raise ValueError("window_seconds and step_seconds must be positive.")

    sampling_frequency = float(session.sampling_frequency)
    window_size = int(round(window_seconds * sampling_frequency))
    step_size = int(round(step_seconds * sampling_frequency))
    if window_size < 2 or step_size < 1:
        raise ValueError("window or step size is too small for the sampling rate.")
    if session.signals.shape[0] < window_size:
        raise ValueError("session is shorter than one analysis window.")

    for name, low, high in frequency_bands:
        if not name:
            raise ValueError("frequency-band names must not be empty.")
        if not 0.0 <= low < high <= sampling_frequency / 2.0:
            raise ValueError("frequency bands must lie within the Nyquist range.")

    feature_names = tuple(
        f"{channel}_{band_name}"
        for channel in session.channel_names
        for band_name, _, _ in frequency_bands
    )
    rows: list[np.ndarray] = []
    times: list[float] = []

    for start in range(0, session.signals.shape[0] - window_size + 1, step_size):
        stop = start + window_size
        window = session.signals[start:stop]
        if not np.isfinite(window).all():
            continue
        frequencies, power = welch(
            window,
            fs=sampling_frequency,
            axis=0,
            nperseg=min(window_size, int(round(sampling_frequency))),
        )
        values: list[float] = []
        for channel_index in range(window.shape[1]):
            for _, low, high in frequency_bands:
                mask = (frequencies >= low) & (frequencies < high)
                band_power = np.trapezoid(
                    power[mask, channel_index],
                    frequencies[mask],
                )
                values.append(float(np.log10(max(band_power, 1e-20))))
        rows.append(np.asarray(values, dtype=float))
        centre = start + (window_size - 1) / 2.0
        times.append(float(session.times[int(round(centre))]))

    if not rows:
        raise ValueError("no finite EEG windows were available for extraction.")
    features = np.vstack(rows)
    time_values = np.asarray(times, dtype=float)
    session_ids = np.full(features.shape[0], session.session, dtype="U8")
    return WindowedFeatureResult(
        features=features,
        times=time_values,
        feature_names=feature_names,
        session_ids=session_ids,
    )


def concatenate_feature_results(
    results: list[WindowedFeatureResult],
) -> WindowedFeatureResult:
    if not results:
        raise ValueError("at least one feature result is required.")
    feature_names = results[0].feature_names
    if any(result.feature_names != feature_names for result in results[1:]):
        raise ValueError("all sessions must contain the same feature columns.")
    features = np.vstack([result.features for result in results])
    session_ids = np.concatenate([result.session_ids for result in results])
    times = np.arange(features.shape[0], dtype=int)
    return WindowedFeatureResult(
        features=features,
        times=times,
        feature_names=feature_names,
        session_ids=session_ids,
    )
