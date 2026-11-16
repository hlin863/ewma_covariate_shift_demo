"""Regression tests for isolated multivariate structural-change scenarios."""
import numpy as np
import pytest

from src.simulation.structural_changes import generate_structural_stream


def get_regimes(scenario, **kwargs):
    _, meta = generate_structural_stream(42, scenario, **kwargs)
    return meta["before"], meta["after"]


def test_only_intended_stationary_target_changes():
    expected = {
        "stationary": set(),
        "mean_shift": {"mean"},
        "variance_shift": {"variance"},
        "correlation_shift": {"correlation"},
        "ar_shift": {"ar"},
        "combined": {"mean", "variance", "correlation", "ar"},
    }
    for scenario, changed in expected.items():
        pre, post = get_regimes(scenario)
        observed = set()
        if pre["mean"] != post["mean"]:
            observed.add("mean")
        b_cov = np.asarray(pre["stationary_covariance"])
        a_cov = np.asarray(post["stationary_covariance"])
        if not np.allclose(np.diag(b_cov), np.diag(a_cov)):
            observed.add("variance")
        b_corr = b_cov[0, 1] / np.sqrt(b_cov[0, 0] * b_cov[1, 1])
        a_corr = a_cov[0, 1] / np.sqrt(a_cov[0, 0] * a_cov[1, 1])
        if not np.isclose(b_corr, a_corr):
            observed.add("correlation")
        if pre["ar"] != post["ar"]:
            observed.add("ar")
        assert observed == changed, scenario


def test_ar_shift_keeps_zero_lag_covariance_but_changes_innovations():
    pre, post = get_regimes("ar_shift")
    np.testing.assert_allclose(pre["stationary_covariance"], post["stationary_covariance"])
    assert pre["ar"] != post["ar"]
    assert not np.allclose(pre["innovation_covariance"], post["innovation_covariance"])
    for p in (pre, post):
        np.testing.assert_allclose(
            p["innovation_covariance"],
            (1 - p["ar"]**2) * np.asarray(p["stationary_covariance"]),
        )


@pytest.mark.parametrize("scenario", [
    "stationary", "mean_shift", "variance_shift",
    "correlation_shift", "ar_shift", "combined",
])
def test_reproducibility_and_metadata(scenario):
    a, m1 = generate_structural_stream(7, scenario, n_observations=256, change_at=128)
    b, m2 = generate_structural_stream(7, scenario, n_observations=256, change_at=128)
    np.testing.assert_array_equal(a, b)
    assert m1 == m2
    assert a.shape == (256, 2)
    assert np.isfinite(a).all()


def test_same_seed_scenarios_are_no_longer_identical():
    streams = [
        generate_structural_stream(42, s, n_observations=100)[0]
        for s in ("mean_shift", "variance_shift", "correlation_shift", "ar_shift", "combined")
    ]
    for i, x in enumerate(streams):
        for y in streams[i + 1:]:
            assert not np.array_equal(x, y)


def test_population_signatures_and_lag_correlation():
    # Long post-change segments separate stationary target properties from
    # short regime-boundary transients and Monte Carlo sampling variability.
    for scenario in ("variance_shift", "correlation_shift", "ar_shift"):
        samples, meta = generate_structural_stream(
            36, scenario, n_observations=22000, change_at=11000
        )
        pre = samples[1000:10000]
        post = samples[12000:21000]
        pre_cov, post_cov = np.cov(pre, rowvar=False), np.cov(post, rowvar=False)
        if scenario == "variance_shift":
            np.testing.assert_allclose(np.diag(post_cov), [4, 4], atol=0.30)
            np.testing.assert_allclose(np.diag(pre_cov), [1, 1], atol=0.12)
            assert abs(np.corrcoef(post.T)[0, 1]) < 0.07
        elif scenario == "correlation_shift":
            np.testing.assert_allclose(np.diag(post_cov), [1, 1], atol=0.12)
            assert abs(np.corrcoef(pre.T)[0, 1]) < 0.07
            assert abs(np.corrcoef(post.T)[0, 1] - 0.7) < 0.07
        else:
            np.testing.assert_allclose(pre_cov, post_cov, atol=0.14)
            lag_pre = np.corrcoef(pre[:-1, 0], pre[1:, 0])[0, 1]
            lag_post = np.corrcoef(post[:-1, 0], post[1:, 0])[0, 1]
            assert abs(lag_pre) < 0.06
            assert abs(lag_post - 0.7) < 0.06


def test_boundary_conventions_and_aliases():
    for index in (0, 40):
        data, metadata = generate_structural_stream(
            9, "mean_shift", n_observations=40, change_at=index
        )
        assert data.shape == (40, 2)
        assert metadata["change_at"] == index
    for alias, expected in (
        ("covariance_shift", "correlation_shift"),
        ("autocorrelation_shift", "ar_shift"),
        ("default", "combined"),
    ):
        x, info = generate_structural_stream(5, alias)
        y, _ = generate_structural_stream(5, expected)
        np.testing.assert_array_equal(x, y)
        assert info["resolved_scenario"] == expected


@pytest.mark.parametrize("kwargs", [
    {"scenario": "typo"}, {"scenario": "combined", "variance_ratio": 0},
    {"scenario": "combined", "variance_ratio": float("nan")},
    {"scenario": "combined", "ar_after": 1},
    {"scenario": "combined", "correlation_after": -1},
    {"scenario": "combined", "change_at": -1},
    {"scenario": "combined", "n_observations": 1},
])
def test_reject_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        generate_structural_stream(1, **kwargs)
