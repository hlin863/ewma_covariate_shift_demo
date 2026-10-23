"""Numerical regression tests for the ABCD equation-level demonstration."""
import math
import pytest

from src.experiments.abcd_lab import ABCDLabConfig, run_abcd_lab


def test_default_is_reproducible_and_structured():
    a = run_abcd_lab(ABCDLabConfig())
    b = run_abcd_lab(ABCDLabConfig())
    assert a == b
    assert len(a["observations"]) == 400
    assert a["summary"]["actual_change_after"] == 200
    assert len(a["splits"]) <= 21
    assert a["summary"]["estimated_split_after"] in [r["k"] for r in a["splits"]]
    assert a["metadata"]["method_status"].startswith("batch")
    assert math.isfinite(a["summary"]["min_raw_bound"])


def test_two_scenarios_and_safe_range():
    for mode in ("Normal-M", "Normal-V"):
        result = run_abcd_lab(ABCDLabConfig(scenario=mode, seed=7))
        assert result["metadata"]["dataset_name"] == mode
        assert all(math.isfinite(v["loss"]) and 0 <= v["loss"] <= 1
                   for v in result["observations"])
        assert all(math.isclose(s["term_before"] + s["term_after"], s["bound"])
                   for s in result["splits"])


@pytest.mark.parametrize("config", [
    {"segment_size": 10000}, {"dimensions": 2}, {"changed_dimensions": 24},
    {"bottleneck": 24}, {"seed": -1}, {"scenario": "HAR"},
    {"shift": float("nan")}, {"delta": 1.1}, {"M": 0.01},
    {"candidate_count": 500},
])
def test_bad_configuration(config):
    with pytest.raises(ValueError):
        run_abcd_lab(ABCDLabConfig(**config))
