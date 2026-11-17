"""KL diagnostic regression tests: no mutation, no labels, fixed calibration bins."""
import numpy as np
import pandas as pd
import pytest

from src.detection.stage2.kl_diagnostics import (
    KLDiagnosticConfig, attach_kl_diagnostics,
    fit_kl_histogram_edges, histogram_kl_scores,
)


def test_identical_windows_have_zero_kl():
    x = np.array([[-2.], [-1.], [0.], [1.], [2.]])
    edges = fit_kl_histogram_edges(x, KLDiagnosticConfig(n_bins=4))
    a, b = histogram_kl_scores(x, x, edges)
    np.testing.assert_allclose(a, 0, atol=1e-12)
    np.testing.assert_allclose(b, 0, atol=1e-12)


def test_shift_scores_positive_without_evaluation_bin_leakage():
    training = np.linspace(-1, 1, 50)[:, None]
    edges = fit_kl_histogram_edges(training)
    a, b = histogram_kl_scores(training, training + 3, edges)
    assert a[0] > 0 and b[0] > 0
    assert np.isneginf(edges[0, 0]) and np.isposinf(edges[0, -1])


def test_attach_preserves_hotelling_decisions_and_skips_pending():
    training = np.arange(40, dtype=float)[:, None]
    testing = np.r_[np.zeros(10), np.ones(10) * 10][:, None]
    validations = pd.DataFrame([
        dict(alarm_time=9, sample_size=10, status="confirmed",
             confirmed_shift=True, p_value=0.001),
        dict(alarm_time=19, sample_size=10, status="pending_current_window",
             confirmed_shift=False, p_value=np.nan),
    ])
    output = attach_kl_diagnostics(validations, testing, np.arange(20), training)
    assert output.loc[0, "kl_reference_to_current"] > 0
    assert output.loc[0, "kl_status"] == "diagnostic_only"
    assert output.loc[1, "kl_status"] == "ineligible_pending_current_window"
    assert np.isnan(output.loc[1, "kl_reference_to_current"])
    pd.testing.assert_frame_equal(
        output[validations.columns].reset_index(drop=True),
        validations.reset_index(drop=True),
    )
    assert "kl_status" not in validations.columns


def test_input_validation_and_constant_training():
    config = KLDiagnosticConfig(n_bins=4)
    edges = fit_kl_histogram_edges(np.ones((12, 2)), config)
    assert edges.shape == (2, 5)
    assert np.all(np.diff(edges, axis=1) > 0)
    with pytest.raises(ValueError):
        KLDiagnosticConfig(pseudocount=0)
    with pytest.raises(ValueError):
        histogram_kl_scores(np.ones((5, 2)), np.ones((5, 3)), edges)
