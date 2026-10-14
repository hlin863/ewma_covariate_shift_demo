import numpy as np
import pytest
from scripts.generate_slide22_structural_visuals import (
    SCENARIOS, generate_isolated_stream, rolling_signatures, make_figures
)


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_seed_shape_and_metadata(scenario):
    x, meta = generate_isolated_stream(scenario, seed=4, n=1024, change_at=512)
    y, _ = generate_isolated_stream(scenario, seed=4, n=1024, change_at=512)
    assert x.shape == (1024,)
    assert np.isfinite(x).all()
    np.testing.assert_array_equal(x, y)
    assert meta["change_at"] == 512


def test_target_parameters_are_isolated():
    for scenario in SCENARIOS:
        _, meta = generate_isolated_stream(scenario)
        b, a = meta["baseline"], meta["after"]
        diff = [i for i in range(3) if b[i] != a[i]]
        assert diff == [{"mean_shift": 0, "variance_shift": 1,
                         "dependence_shift": 2}[scenario]]


def test_rolling_signatures_dimensions():
    x, _ = generate_isolated_stream("variance_shift", n=128, change_at=64)
    t, stats = rolling_signatures(x, window=20)
    assert stats.shape == (3, 109)
    assert t[0] == 19 and t[-1] == 127


def test_invalid_parameters():
    with pytest.raises(ValueError):
        generate_isolated_stream("unrecognised")
    with pytest.raises(ValueError):
        generate_isolated_stream("mean_shift", n=4)
    with pytest.raises(ValueError):
        rolling_signatures(np.arange(10), window=11)


def test_expected_population_signatures_are_distinct():
    expected = {
        "mean_shift": (1.5, 1.0, 0.25),
        "variance_shift": (0.0, 4.0, 0.25),
        "dependence_shift": (0.0, 1.0, 0.75),
    }
    for name, params in expected.items():
        _, metadata = generate_isolated_stream(name)
        assert metadata["after"] == params


def test_figure_outputs(tmp_path):
    make_figures(tmp_path, seed=42, n=128, change_at=64, window=24)
    for name in ("scenarios", "rolling_signatures", "method_targets"):
        assert (tmp_path / f"slide22_{name}.png").stat().st_size > 0
        assert (tmp_path / f"slide22_{name}.svg").stat().st_size > 0
    assert (tmp_path / "config.json").exists()
