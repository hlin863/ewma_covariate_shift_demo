"""Ensure measured on-line comparison reports consistent causal outcomes."""
import numpy as np
import pytest
from src.online.experiments import run_online_comparison


def test_policy_comparison_has_control_and_genuine_measures():
    result = run_online_comparison(
        scenario="feature_shift", seed=12, trials=80, window_size=5,
        label_delay=1, interval=10,
    )
    summary = {row["policy"]:row for row in result["summary"]}
    assert set(summary) == {"never","periodic","validated"}
    assert summary["never"]["updates"] == 0
    assert summary["never"]["gain_vs_never"] == 0.0
    assert summary["periodic"]["updates"] > 0
    assert all(0.0 <= row["accuracy"] <= 1.0 for row in summary.values())
    for trace in result["traces"].values():
        assert len(trace) == 80
        assert [item["time"] for item in trace] == list(range(80))


def test_labels_withheld_prevents_classifier_updates():
    result = run_online_comparison(
        scenario="none", seed=10, trials=80, label_delay=None, window_size=5,
    )
    assert all(row["updates"] == 0 for row in result["summary"])


@pytest.mark.parametrize("overrides", [
    {"trials":10000}, {"scenario":"random"}, {"window_size":0},
    {"label_delay":-2}, {"interval":0},
])
def test_invalid_comparison_inputs_rejected(overrides):
    with pytest.raises(ValueError):
        run_online_comparison(**overrides)
