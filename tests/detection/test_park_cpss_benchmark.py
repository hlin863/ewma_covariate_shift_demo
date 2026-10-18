"""The benchmark extends (rather than replaces) the published Park rule."""
import json

import numpy as np
import pytest

from src.experiments.park2023.cpss_benchmark import (
    BenchmarkConfig, SCENARIOS, binary_metrics, calibrate_threshold,
    evaluate_trial, export_benchmark, make_stream, roc_auc, run_benchmark,
)


def small_config(**overrides):
    params = {
        "observations": 120,
        "channels": 3,
        "calibration_seeds": 2,
        "evaluation_seeds": 2,
        "perturbations": 2,
        "seed": 19,
    }
    params.update(overrides)
    return BenchmarkConfig(**params)


def test_generator_scenarios_are_paired_and_mean_shift_occurs_at_midpoint():
    config = small_config()
    null = make_stream("stationary", config, 23)
    shifted = make_stream("mean_shift", config, 23)
    variance = make_stream("variance_shift", config, 23)
    midpoint = config.observations // 2
    assert set(SCENARIOS) == {"stationary", "mean_shift", "variance_shift"}
    assert np.array_equal(null[:midpoint], shifted[:midpoint])
    assert np.array_equal(null[:midpoint], variance[:midpoint])
    assert np.array_equal(null[midpoint:, -1], shifted[midpoint:, -1])
    assert np.allclose(shifted[midpoint:, 0] - null[midpoint:, 0],
                       config.mean_shift_size)
    assert np.all(null > 0)


def test_zero_jitter_zero_gamma_recovers_original_binary_decision():
    config = small_config(perturbations=3, jitter_sd=0.0, gamma=0.0)
    for scenario in SCENARIOS:
        trial = evaluate_trial(make_stream(scenario, config, 23), config, 777)
        assert trial["cpss_score"] == pytest.approx(trial["binary_decision"])
        assert trial["contribution_std"] == pytest.approx(0.0)


def test_empirical_threshold_uses_strict_greater_than_rule():
    assert calibrate_threshold([0, 0, 0, 0], 0.05) == 0.0
    assert calibrate_threshold([0.1, 0.2, 0.3, 0.4], 0.25) == 0.4
    with pytest.raises(ValueError):
        calibrate_threshold([], 0.05)


def test_tie_aware_auc_and_binary_metrics():
    assert roc_auc([0.0, 0.0, 0.0, 0.0], [0, 0, 1, 1]) == 0.5
    assert roc_auc([0.1, 0.2, 0.9, 1.0], [0, 0, 1, 1]) == 1.0
    outcome = binary_metrics([0, 0, 1, 1], [0, 1, 0, 1])
    assert outcome["accuracy"] == 0.5
    assert outcome["balanced_accuracy"] == 0.5
    assert outcome["tp"] == outcome["fp"] == 1


def test_paired_benchmark_writes_truth_and_spread_outputs(tmp_path):
    config = small_config()
    result = run_benchmark(config)
    assert len(result["rows"]) == config.evaluation_seeds * len(SCENARIOS)
    assert result["calibration"]["seed_range"][1] < result["evaluation_seed_range"][0]
    assert len(result["calibration"]["scores"]) == config.calibration_seeds
    assert set(result["spread"]) == set(SCENARIOS)
    for name in ("binary", "cpss"):
        stats = result["classification"][name]
        assert stats["tp"] + stats["fn"] == config.evaluation_seeds
        assert stats["tn"] + stats["fp"] == config.evaluation_seeds
        assert 0 <= stats["accuracy"] <= 1
        assert 0 <= stats["roc_auc"] <= 1
        assert 0 <= stats["variance_only_positive_rate"] <= 1
    assert all(0 <= row["cpss_score"] <= 1 for row in result["rows"])
    assert all(row["binary_absolute_position_error"] is None
               for row in result["rows"] if row["scenario"] == "stationary")
    saved = export_benchmark(result, tmp_path / "park_run")
    summary = json.loads((tmp_path / "park_run" / "summary.json").read_text())
    assert summary["config"]["seed"] == config.seed
    assert "rows" not in summary
    assert (tmp_path / "park_run" / "trials.csv").exists()
    with pytest.raises(FileExistsError):
        export_benchmark(result, tmp_path / "park_run")


@pytest.mark.parametrize("overrides", [
    {"observations": 80}, {"channels": 1}, {"perturbations": 0},
    {"target_fpr": 0}, {"jitter_sd": -0.1}, {"gamma": -1},
])
def test_invalid_config_rejected(overrides):
    with pytest.raises(ValueError):
        small_config(**overrides)
