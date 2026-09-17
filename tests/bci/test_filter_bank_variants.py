"""Tests for the two paper-specific Butterworth/FBCSP filter-bank variants."""

import numpy as np

from src.bci.fbcsp import (
    CSE_UAEL_FILTER_BANK,
    ONLINE_BCI_2018_FILTER_BANK,
    PAPER_FILTER_BANK,
    fit_transform_fbcsp,
)


def _synthetic_trials() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    sampling_frequency = 250.0
    time = np.arange(256) / sampling_frequency
    labels = np.array([0, 1] * 6)
    trials = rng.normal(0.0, 0.15, size=(labels.size, 4, time.size))

    for index, label in enumerate(labels):
        if label == 0:
            trials[index, 0] += np.sin(2.0 * np.pi * 10.0 * time)
            trials[index, 1] += 0.5 * np.sin(2.0 * np.pi * 18.0 * time)
        else:
            trials[index, 2] += np.sin(2.0 * np.pi * 10.0 * time)
            trials[index, 3] += 0.5 * np.sin(2.0 * np.pi * 18.0 * time)
    return trials, labels


def test_filter_bank_definitions_preserve_both_methodological_variants() -> None:
    assert len(CSE_UAEL_FILTER_BANK) == 10
    assert CSE_UAEL_FILTER_BANK[0] == (8.0, 12.0)
    assert CSE_UAEL_FILTER_BANK[-1] == (26.0, 30.0)
    assert ONLINE_BCI_2018_FILTER_BANK == ((8.0, 12.0), (16.0, 24.0))
    assert PAPER_FILTER_BANK is CSE_UAEL_FILTER_BANK


def test_cse_uael_filter_bank_produces_twenty_features() -> None:
    trials, labels = _synthetic_trials()
    model, training_features, testing_features = fit_transform_fbcsp(
        trials,
        labels,
        trials[:4],
        250.0,
        filter_bank=CSE_UAEL_FILTER_BANK,
    )

    assert len(model.models) == 10
    assert model.n_features == 20
    assert training_features.shape == (12, 20)
    assert testing_features.shape == (4, 20)


def test_online_bci_2018_filter_bank_produces_four_features() -> None:
    trials, labels = _synthetic_trials()
    model, training_features, testing_features = fit_transform_fbcsp(
        trials,
        labels,
        trials[:4],
        250.0,
        filter_bank=ONLINE_BCI_2018_FILTER_BANK,
    )

    assert tuple((band.low_hz, band.high_hz) for band in model.models) == (
        (8.0, 12.0),
        (16.0, 24.0),
    )
    assert model.n_features == 4
    assert training_features.shape == (12, 4)
    assert testing_features.shape == (4, 4)


def test_fit_transform_default_remains_cse_uael_for_existing_callers() -> None:
    trials, labels = _synthetic_trials()
    model, training_features, _ = fit_transform_fbcsp(
        trials,
        labels,
        trials[:2],
        250.0,
    )

    assert len(model.models) == 10
    assert training_features.shape[1] == 20
