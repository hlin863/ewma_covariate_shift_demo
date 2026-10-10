"""Pages-export compatibility regression for archived Diethe results."""
import json
from pathlib import Path

from scripts.build_github_pages import ROOT, compatible_diethe_snapshot


def test_saved_five_policy_diethe_snapshot_is_historical_not_six_policy():
    snapshots = sorted((ROOT / "outputs" / "diethe").glob("*/experiment.json"))
    assert snapshots, "Expected a committed historical Diethe fixture"
    legacy = json.loads(snapshots[-1].read_text(encoding="utf-8"))
    assert "evidence_value" not in legacy["runs"]
    assert compatible_diethe_snapshot(legacy) is False


def test_six_policy_diethe_snapshot_requires_decisions_and_new_summary_fields():
    six = {
        "metadata": {"config": {
            "interval": 20, "performance_window": 20, "update_scope": "current_trial"
        }},
        "runs": {"evidence_value": {"evidence_decisions": []}},
        "summary": [{"policy_family": "active", "trigger": "evidence"}],
    }
    assert compatible_diethe_snapshot(six) is True
    six["runs"]["evidence_value"].pop("evidence_decisions")
    assert compatible_diethe_snapshot(six) is False
