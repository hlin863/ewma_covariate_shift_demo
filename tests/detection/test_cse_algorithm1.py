import numpy as np
import pandas as pd

from src.cse import CSEConfig, run_cse
from src.cse_algorithm1_stage_2 import (
    calculate_training_reference_hotelling,
    validate_algorithm1_alarms,
)


def test_algorithm1_hotelling_detects_distant_feature_vector() -> None:
    rng = np.random.default_rng(41)
    training = rng.normal(0.0, 0.25, size=(120, 3))
    current = np.array([3.0, 3.0, 3.0])

    _, _, _, _, p_value = calculate_training_reference_hotelling(
        current,
        training,
    )

    assert p_value < 0.05


def test_algorithm1_validator_evaluates_every_warning_immediately() -> None:
    rng = np.random.default_rng(42)
    training = rng.normal(0.0, 0.3, size=(100, 2))
    testing = np.vstack([
        rng.normal(0.0, 0.3, size=(3, 2)),
        np.array([[4.0, 4.0], [5.0, 5.0]]),
    ])
    times = np.arange(testing.shape[0])
    stage_1 = pd.DataFrame({
        "time": times,
        "stage_1_alarm": [0, 0, 0, 1, 1],
    })

    results = validate_algorithm1_alarms(
        training_features=training,
        testing_features=testing,
        testing_times=times,
        stage_1_results=stage_1,
    )

    assert results.shape[0] == 2
    assert results["confirmed_shift"].all()
    np.testing.assert_array_equal(
        results["validation_time"].to_numpy(),
        results["alarm_time"].to_numpy(),
    )


def test_algorithm1_cse_starts_testing_ewma_from_training_mean() -> None:
    rng = np.random.default_rng(43)
    training = rng.normal(0.0, 1.0, size=(80, 4))
    testing = rng.normal(0.0, 1.0, size=(20, 4))

    result = run_cse(
        training_features=training,
        testing_features=testing,
        testing_times=np.arange(testing.shape[0]),
        config=CSEConfig(
            pca_components=3,
            lambda_override=0.28,
            ewma_initialization="training_mean",
            validation_mode="algorithm1_training_reference",
        ),
    )

    first = result.warning_results.iloc[0]
    assert np.isclose(first["prediction"], result.ewma_training_result.initial_z)
    assert result.effective_lambda == 0.28
    assert "training_size" in result.validation_results.columns
