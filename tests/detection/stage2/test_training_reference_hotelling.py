import numpy as np
import pandas as pd

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
