import numpy as np
import pytest

from src.ewma import SD_EWMA_Config
from src.stage_2 import Stage2Config
from src.tssd_ewma import (
    TSSDEWMAConfig,
    run_tssd_ewma,
)


def test_tssd_pipeline_returns_all_result_components():
    rng = np.random.default_rng(42)

    training_values = rng.normal(
        loc=0.0,
        scale=1.0,
        size=200,
    )

    testing_values = np.concatenate([
        rng.normal(0.0, 1.0, 100),
        rng.normal(5.0, 1.0, 100),
    ])

    testing_times = np.arange(testing_values.size)

    config = TSSDEWMAConfig(
        stage_1=SD_EWMA_Config(
            lambda_value=0.5,
            variance_smoothing=0.05,
            control_limit_multiplier=3.0,
        ),
        stage_2=Stage2Config(
            before_size=20,
            after_size=20,
            alpha=0.05,
            minimum_alarm_gap=20,
        ),
    )

    result = run_tssd_ewma(
        training_values=training_values,
        testing_values=testing_values,
        testing_times=testing_times,
        config=config,
    )

    assert result.training_result is not None
    assert not result.stage_1_results.empty
    assert "stage_1_alarm" in result.stage_1_results.columns
    assert "status" in result.stage_2_results.columns
    assert result.computation_time_seconds >= 0.0


def test_tssd_pipeline_rejects_mismatched_testing_inputs():
    config = TSSDEWMAConfig(
        stage_1=SD_EWMA_Config(
            lambda_value=0.5,
        ),
        stage_2=Stage2Config(),
    )

    with pytest.raises(
        ValueError,
        match="must have equal length",
    ):
        run_tssd_ewma(
            training_values=np.zeros(20),
            testing_values=np.zeros(10),
            testing_times=np.arange(9),
            config=config,
        )

def test_configured_lambda_and_variance_mode_are_preserved() -> None:
    rng = np.random.default_rng(4)
    training = rng.normal(0.0, 1.0, 100)
    testing = np.concatenate([rng.normal(0.0, 1.0, 50), rng.normal(4.0, 1.0, 50)])
    result = run_tssd_ewma(
        training,
        testing,
        np.arange(100, 200),
        TSSDEWMAConfig(
            stage_1=SD_EWMA_Config(
                lambda_value=0.4,
                variance_update_mode="frozen",
            ),
            stage_2=Stage2Config(before_size=10, after_size=10),
            lambda_mode="configured",
        ),
    )

    assert result.training_result.lambda_value == 0.4
    assert set(result.stage_1_results["variance_update_mode"]) == {"frozen"}
