import numpy as np
import pytest

from src.detection.baselines.ici_cdt import (
    ICICDTConfig,
    fit_ici_cdt,
    run_ici_cdt,
)


def test_fit_ici_cdt_requires_two_complete_blocks() -> None:
    with pytest.raises(ValueError, match="two complete blocks"):
        fit_ici_cdt(np.arange(10.0), ICICDTConfig(block_size=10))


def test_ici_cdt_detects_a_sustained_mean_change() -> None:
    rng = np.random.default_rng(4)
    training = rng.normal(0.0, 0.2, 200)
    testing = np.concatenate(
        [rng.normal(0.0, 0.2, 100), rng.normal(3.0, 0.2, 100)]
    )
    result = run_ici_cdt(
        training,
        testing,
        np.arange(201, 401),
        ICICDTConfig(block_size=10, confidence_parameter=2.0),
    )

    alarm_times = result.trace.loc[result.trace["ici_alarm"] == 1, "time"]
    assert not alarm_times.empty
    assert int(alarm_times.iloc[0]) >= 301
    assert result.computation_time_seconds >= 0.0


def test_ici_cdt_validates_config_and_stream_shapes() -> None:
    with pytest.raises(ValueError, match="block_size"):
        ICICDTConfig(block_size=0)
    with pytest.raises(ValueError, match="match testing_values"):
        run_ici_cdt(np.arange(20.0), np.arange(20.0), np.arange(19))
