import numpy as np
import pandas as pd
import pytest

from src.simulation import JumpingMeanConfig, generate_d2_jumping_mean


def test_public_simulation_namespace_exports_d2_generator() -> None:
    assert callable(generate_d2_jumping_mean)


def test_default_generator_has_paper_shape_and_truth_columns() -> None:
    stream = generate_d2_jumping_mean()

    assert stream.shape == (5000, 6)
    assert list(stream.columns) == [
        "time",
        "x",
        "innovation",
        "noise_mean",
        "regime",
        "true_shift",
    ]
    assert stream.loc[0, "time"] == 1
    assert stream.loc[4999, "time"] == 5000
    assert stream.loc[0, "x"] == 0.0
    assert stream.loc[1, "x"] == 0.0
    assert int(stream["true_shift"].sum()) == 49


def test_regime_boundaries_use_paper_one_based_time() -> None:
    stream = generate_d2_jumping_mean()

    shift_times = stream.loc[stream["true_shift"] == 1, "time"].tolist()
    assert shift_times[:3] == [101, 201, 301]
    assert shift_times[-1] == 4901
    assert stream.loc[99, "regime"] == 1
    assert stream.loc[100, "regime"] == 2


def test_noise_mean_follows_cumulative_paper_schedule() -> None:
    stream = generate_d2_jumping_mean()

    assert stream.loc[0, "noise_mean"] == 0.0
    assert stream.loc[100, "noise_mean"] == pytest.approx(2 / 16)
    assert stream.loc[200, "noise_mean"] == pytest.approx(2 / 16 + 3 / 16)
    assert stream.loc[4900, "noise_mean"] == pytest.approx(
        sum(range(2, 51)) / 16
    )


def test_generated_values_satisfy_ar_recurrence() -> None:
    config = JumpingMeanConfig(random_seed=7)
    stream = generate_d2_jumping_mean(config)
    values = stream["x"].to_numpy()
    innovations = stream["innovation"].to_numpy()

    reconstructed = (
        config.ar_coefficient_1 * values[1:-1]
        + config.ar_coefficient_2 * values[:-2]
        + innovations[2:]
    )
    np.testing.assert_allclose(values[2:], reconstructed)


def test_generator_is_reproducible_without_global_rng_state() -> None:
    first = generate_d2_jumping_mean(JumpingMeanConfig(random_seed=17))
    second = generate_d2_jumping_mean(JumpingMeanConfig(random_seed=17))
    different = generate_d2_jumping_mean(JumpingMeanConfig(random_seed=18))

    pd.testing.assert_frame_equal(first, second)
    assert not np.array_equal(first["x"].to_numpy(), different["x"].to_numpy())


@pytest.mark.parametrize(
    "config",
    [
        JumpingMeanConfig(n_samples=2),
        JumpingMeanConfig(shift_interval=1),
    ],
)
def test_valid_edge_configurations_generate(config: JumpingMeanConfig) -> None:
    stream = generate_d2_jumping_mean(config)
    assert len(stream) == config.n_samples


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_samples": 1}, "n_samples"),
        ({"shift_interval": 0}, "shift_interval"),
        ({"noise_std": 0.0}, "noise_std"),
        ({"ar_coefficient_1": np.nan}, "finite"),
    ],
)
def test_invalid_configuration_raises(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        JumpingMeanConfig(**kwargs)
