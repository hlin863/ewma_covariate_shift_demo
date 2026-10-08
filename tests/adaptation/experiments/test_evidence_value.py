import pytest
from src.adaptation.evidence_value import (
    DecisionAction, EvidenceDecisionContext, EvidenceValuePolicy,
)
from src.adaptation.experiments.diethe import DietheConfig, run_synthetic


def context(accuracy=.7, n=30, elapsed=30):
    return EvidenceDecisionContext(accuracy, .9, n, elapsed)


def test_high_cost_prefers_immediate_update():
    policy = EvidenceValuePolicy(10, .2, .2, 1.0, .1, 5)
    result = policy.decide(context())
    assert result.action == DecisionAction.UPDATE_NOW
    assert result.net_collection_value < 0


def test_low_cost_can_prefer_collecting_more():
    policy = EvidenceValuePolicy(10, 0, 0, 10, .1, 5)
    result = policy.decide(context())
    assert result.action == DecisionAction.COLLECT_MORE_EVIDENCE
    assert result.projected_standard_error < result.current_standard_error
    assert result.net_collection_value > 0


def test_insufficient_deficit_no_update():
    policy = EvidenceValuePolicy()
    assert policy.decide(context(.88)).action == DecisionAction.NO_UPDATE


def test_wait_cooldown_and_cost_validation():
    policy = EvidenceValuePolicy(10, 0, 0, 10, .1, 5)
    assert policy.decide(context(.7, elapsed=3)).action == DecisionAction.NO_UPDATE
    assert policy.decide(EvidenceDecisionContext(None, .9, 30, 30)).action == DecisionAction.COLLECT_MORE_EVIDENCE
    with pytest.raises(ValueError):
        EvidenceValuePolicy(cost_per_sample=-1)
    with pytest.raises(ValueError):
        EvidenceValuePolicy(additional_samples=0)


def test_diethe_six_policies_and_evidence_export_contract():
    result = run_synthetic(DietheConfig(seed=42, trials=100, evidence_samples=5))
    assert len(result["summary"]) == 6
    assert "evidence_value" in result["runs"]
    events = result["runs"]["evidence_value"]["evidence_decisions"]
    assert len(events) == 100
    assert {e["action"] for e in events} <= {
        "update_now", "no_update", "collect_more_evidence"
    }
    assert len(result["runs"]["evidence_value"]["trials"]) == 100


def test_cost_sensitivity_comparison():
    low = run_synthetic(DietheConfig(seed=42, trials=100,
                                     evidence_sample_cost=0, evidence_delay_cost=0,
                                     evidence_consequence_weight=10))
    high = run_synthetic(DietheConfig(seed=42, trials=100,
                                      evidence_sample_cost=2, evidence_delay_cost=2))
    low_events = low["runs"]["evidence_value"]["evidence_decisions"]
    high_events = high["runs"]["evidence_value"]["evidence_decisions"]
    assert sum(e["action"] == "collect_more_evidence" for e in low_events) >= (
        sum(e["action"] == "collect_more_evidence" for e in high_events)
    )
    assert [r["policy"] for r in low["summary"]][:5] == [
        "never", "periodic", "warning", "validated", "performance_drop"
    ]
