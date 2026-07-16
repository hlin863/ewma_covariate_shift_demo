import numpy as np
import pytest

from src.ewma import (
    SD_EWMA_Config,
    fit_sd_ewma,
    run_sd_ewma,
)


@pytest.mark.integration
def test_d1_detects_a_post_shift_alarm(
    d1_generator,
):
    stream = d1_generator(seed=42)

    training_values = stream[:500]
    testing_values = stream[500:]
    testing_times = np.arange(500, 2000)

    training_result = fit_sd_ewma(
        training_values
    )

    results = run_sd_ewma(
        values=testing_values,
        times=testing_times,
        initial_z=training_result.final_z,
        initial_error_variance=(
            training_result.error_variance
        ),
        config=SD_EWMA_Config(
            lambda_value=(
                training_result.lambda_value
            ),
            variance_smoothing=0.05,
            control_limit_multiplier=3.0,
        ),
    )

    post_shift_alarms = results.loc[
        (results["time"] >= 1000)
        & (results["stage_1_alarm"] == 1)
    ]

    assert not post_shift_alarms.empty

    first_alarm = int(
        post_shift_alarms.iloc[0]["time"]
    )

    assert first_alarm <= 1020