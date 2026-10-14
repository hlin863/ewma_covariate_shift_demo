import numpy as np
import pytest
from src.simulation.park_structural_models import simulate_park_model


@pytest.mark.parametrize("model", ["A", "B", "C"])
@pytest.mark.parametrize("change_at", [None, 120])
def test_shape_seed_and_affected_channels(model, change_at):
    a = simulate_park_model(model, n=240, channels=10, change_at=change_at, seed=42)
    b = simulate_park_model(model, n=240, channels=10, change_at=change_at, seed=42)
    assert a["values"].shape == (240, 10)
    assert np.isfinite(a["values"]).all()
    np.testing.assert_array_equal(a["values"], b["values"])
    assert len(a["affected_channels"]) == (5 if change_at is not None else 0)


@pytest.mark.parametrize("model", ["A", "B", "C"])
def test_seed_changes_draws(model):
    assert not np.array_equal(simulate_park_model(model, seed=1)["values"],
                              simulate_park_model(model, seed=2)["values"])


@pytest.mark.parametrize("kwargs", [
    {"model": "D"}, {"model": "A", "n": 3},
    {"model": "A", "channels": 1}, {"model": "A", "change_at": 1024},
    {"model": "A", "rho": 1.2}
])
def test_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        simulate_park_model(**kwargs)


def test_mean_shift_in_model_a():
    x = simulate_park_model("A", n=5000, channels=10, change_at=2500,
                            rho=1.0, seed=12)["values"]
    assert x[2600:].mean() > x[200:2400].mean() + 1


@pytest.mark.parametrize("model", ["B", "C"])
def test_second_order_zero_mean(model):
    x = simulate_park_model(model, n=5000, channels=20, change_at=2500,
                            rho=1.0, seed=12)["values"]
    assert abs(x.mean()) < 0.12
