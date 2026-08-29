import numpy as np
import pandas as pd

from src.detection.multivariate_preprocessing import fit_paper_ica, fit_paper_pca
from src.detection.stage1.paper_msd_ewma import (
    fit_paper_msd_ewma,
    run_paper_msd_ewma,
)
from src.detection.stage2.paper_hotelling import (
    PaperHotellingConfig,
    calculate_two_sample_hotelling,
    validate_paper_hotelling_alarms,
)


def test_paper_msd_ewma_uses_equation_9_covariance_scale() -> None:
    training_values = np.array([[-1.0], [1.0]])
    fitted = fit_paper_msd_ewma(training_values, lambda_value=0.1, regularization=0.0)
    result = run_paper_msd_ewma(
        np.array([[2.0], [2.0]]),
        np.array([1, 2]),
        fitted,
        control_limit=10.0,
    )

    first_z = 0.1 * 2.0
    covariance = np.var(training_values[:, 0], ddof=1)
    scale = 0.1 / 1.9 * (1.0 - 0.9**2)
    expected = first_z**2 / (scale * covariance)
    assert np.isclose(result.loc[0, "t_squared"], expected)


def test_pca_and_ica_are_fit_on_training_then_applied_to_full_stream() -> None:
    rng = np.random.default_rng(4)
    training = rng.normal(size=(100, 3))
    full = np.vstack([training, rng.normal(loc=2.0, size=(10, 3))])

    pca = fit_paper_pca(training, full, n_components=2)
    ica = fit_paper_ica(training, full, random_seed=4)

    assert pca.training_transformed.shape == (100, 2)
    assert pca.full_transformed.shape == (110, 2)
    assert ica.training_transformed.shape == (100, 3)
    assert ica.full_transformed.shape == (110, 3)


def test_hotelling_validator_reports_after_the_future_window() -> None:
    rng = np.random.default_rng(5)
    features = np.vstack(
        [rng.normal(size=(30, 2)), rng.normal(loc=3.0, size=(30, 2))]
    )
    times = np.arange(1, 61)
    alarms = pd.DataFrame({"time": times, "stage_1_alarm": times == 30})
    result = validate_paper_hotelling_alarms(
        features,
        times,
        alarms,
        PaperHotellingConfig(window_size=10, alpha=0.05),
    )

    assert result.loc[0, "validation_time"] == 40
    assert bool(result.loc[0, "confirmed_shift"])
    statistic = calculate_two_sample_hotelling(features[20:30], features[30:40])
    assert statistic[-1] < 0.05
