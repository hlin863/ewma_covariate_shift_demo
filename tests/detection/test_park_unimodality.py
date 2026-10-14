import numpy as np
import pytest

from src.detection.stage1.park_mean_cusum import (
    check_cusum_unimodality, evaluate_mean_change, mean_cusum_curves,
)


def test_smooth_single_peak_has_no_shape_violations():
    positions = np.arange(1, 120)
    values = 1 - np.abs(positions - 60) / 60
    result = check_cusum_unimodality(
        positions, values, peak=60, observations=121
    )
    assert result["eligible"]
    assert result["A"] == 0
    assert result["unimodal"]


def test_local_reversal_is_counted_not_mistaken_for_unimodal():
    positions = np.arange(1, 120)
    values = 1 - np.abs(positions - 60) / 60
    # Inject strong decrease in the later half of the rising side,
    # sampled by D1 at approximately 32/38.
    values[positions == 32] = 5
    values[positions == 38] = 4
    # Explicitly pass the desired paper peak to inspect just the shape.
    result = check_cusum_unimodality(
        positions, values, peak=60, observations=121
    )
    assert result["A"] > 0


def test_strict_peak_agreement_rule():
    positions = np.arange(1, 120)
    def peak_at(point):
        return (1 - np.abs(positions - point) / 120).tolist()
    base = {"positions": positions.tolist(), "observations": 121}
    close = evaluate_mean_change(dict(base,
        maximum_smoothed=peak_at(60), average_smoothed=peak_at(61)))
    assert close["mean_change"] is True
    assert close["estimated_position"] == 60.5
    assert close["agreement_gap"] == 1
    far = evaluate_mean_change(dict(base,
        maximum_smoothed=peak_at(52), average_smoothed=peak_at(80)))
    assert far["mean_change"] is False
    assert far["next_step"] == "second_order_procedure_not_implemented"


def test_restricted_domain_can_make_check_ineligible():
    positions = np.arange(8, 10)
    outcome = check_cusum_unimodality(
        positions, np.array([1.0, 2.0]), observations=18)
    assert outcome["eligible"] is False
    assert outcome["A"] is None


def test_missing_fields_and_invalid_k_rejected():
    with pytest.raises(ValueError):
        evaluate_mean_change({"positions": [1]})
    with pytest.raises(ValueError):
        check_cusum_unimodality([1, 2], [1.0, 2.0], observations=8, k=7)


def test_integration_with_existing_mean_cusum():
    rng = np.random.default_rng(15)
    x = rng.normal(loc=10, scale=0.7, size=(240, 3))
    x[120:] += [3, 2, 4]
    curves = mean_cusum_curves(x)
    result = evaluate_mean_change(curves)
    assert result["status"] in {
        "mean_change_candidate", "no_mean_change_by_rule", "ineligible"
    }
    assert "maximum_check" in result
    assert "average_check" in result
