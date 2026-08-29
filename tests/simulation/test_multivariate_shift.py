import numpy as np

from src.simulation.multivariate_shift import (
    MultivariateShiftConfig,
    generate_d3_multivariate_normal,
    generate_d4_multivariate_t,
    paper_covariance_matrix,
)


def test_d3_has_the_paper_shape_covariance_and_shift_markers() -> None:
    config = MultivariateShiftConfig(random_seed=7)
    stream = generate_d3_multivariate_normal(config)

    assert stream.shape == (300, 15)
    assert stream.filter(regex=r"^x\d+$").shape == (300, 10)
    assert stream.loc[stream["true_shift"] == 1, "time"].tolist() == [101, 201]
    assert stream.groupby("regime")["target_mean"].first().tolist() == [0.0, 1.0, 0.0]
    covariance = paper_covariance_matrix(config)
    assert np.allclose(np.diag(covariance), 0.45)
    assert np.allclose(covariance[~np.eye(10, dtype=bool)], 0.30)


def test_d3_and_d4_are_reproducible_and_distribution_specific() -> None:
    config = MultivariateShiftConfig(random_seed=19)
    d3_first = generate_d3_multivariate_normal(config)
    d3_second = generate_d3_multivariate_normal(config)
    d4_first = generate_d4_multivariate_t(config)
    d4_second = generate_d4_multivariate_t(config)

    assert d3_first.equals(d3_second)
    assert d4_first.equals(d4_second)
    assert d3_first["distribution"].unique().tolist() == ["normal"]
    assert d4_first["distribution"].unique().tolist() == ["student_t"]
    assert not np.allclose(
        d3_first.filter(regex=r"^x\d+$"),
        d4_first.filter(regex=r"^x\d+$"),
    )


def test_d4_scale_adjustment_recovers_the_stated_covariance() -> None:
    config = MultivariateShiftConfig(
        random_seed=123,
        n_samples=20_000,
        initial_mean=0.0,
        shifted_mean=0.0,
    )
    values = generate_d4_multivariate_t(config).filter(regex=r"^x\d+$").to_numpy()
    empirical = np.cov(values, rowvar=False)

    assert np.allclose(empirical, paper_covariance_matrix(config), atol=0.04)
