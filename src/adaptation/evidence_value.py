"""Experimental evidence-sufficiency policy for the Diethe laboratory.

A *precision-gain proxy*, NOT the A-value from Epanomeritakis & Viviano
(2026). Their minimax-regret welfare quantities cannot be inferred from a
Hotelling p-value or EEG accuracy alone.

The policy uses the rolling labelled error difference against the held-out
validation reference. Additional sampling reduces its estimated standard error;
the reduction is compared against user-supplied sample and delay costs.
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from math import isfinite, sqrt
from typing import Protocol

class DecisionAction(str, Enum):
    UPDATE_NOW = "update_now"
    COLLECT_MORE_EVIDENCE = "collect_more_evidence"
    NO_UPDATE = "no_update"

@dataclass(frozen=True)
class EvidenceDecisionContext:
    recent_accuracy: float | None
    reference_accuracy: float
    observed_samples: int
    trials_since_update: int

@dataclass(frozen=True)
class EvidenceDecision:
    action: DecisionAction
    performance_deficit: float | None
    current_standard_error: float | None
    projected_standard_error: float | None
    precision_value: float
    collection_cost: float
    delay_cost: float
    net_collection_value: float

class EvidenceActionPolicy(Protocol):
    def decide(self, context: EvidenceDecisionContext) -> EvidenceDecision: ...

@dataclass(frozen=True)
class EvidenceValuePolicy:
    """Heuristic policy; costs are dimensionless accuracy-equivalent units.

    cost_per_sample: evidence-acquisition cost per *new* labelled sample
    cost_per_delay_trial: opportunity cost per trial while awaiting evidence
    consequence_weight: conversion of uncertainty reduction into policy value
    additional_samples: requested additional labelled samples
    min_deficit: performance deficit needed for an immediate update decision
    min_trials_between_updates: minimum cooldown before updating
    """
    additional_samples: int = 10
    cost_per_sample: float = 0.001
    cost_per_delay_trial: float = 0.001
    consequence_weight: float = 1.0
    min_deficit: float = 0.10
    min_trials_between_updates: int = 5

    def __post_init__(self):
        if self.additional_samples < 1 or self.min_trials_between_updates < 1:
            raise ValueError("Sample budget and cooldown must be positive.")
        for field in ("cost_per_sample", "cost_per_delay_trial", "consequence_weight", "min_deficit"):
            value = getattr(self, field)
            if not isfinite(value) or value < 0:
                raise ValueError(f"{field} must be finite and nonnegative.")
        if self.min_deficit > 1:
            raise ValueError("min_deficit must not exceed one.")

    def decide(self, context: EvidenceDecisionContext) -> EvidenceDecision:
        observed = context.observed_samples
        accuracy = context.recent_accuracy
        if observed < 1 or context.trials_since_update < 0:
            raise ValueError("Observed samples must be positive and update age nonnegative.")
        if not 0 <= context.reference_accuracy <= 1:
            raise ValueError("Reference accuracy must be in [0,1].")
        cost = self.additional_samples * self.cost_per_sample
        delay_cost = self.additional_samples * self.cost_per_delay_trial
        if accuracy is None:
            return EvidenceDecision(DecisionAction.COLLECT_MORE_EVIDENCE, None, None, None,
                                    0.0, cost, delay_cost, -(cost + delay_cost))
        if not isfinite(accuracy) or not 0 <= accuracy <= 1:
            raise ValueError("Recent accuracy must be finite in [0,1].")
        deficit = max(0.0, context.reference_accuracy - accuracy)
        # Fixed validation accuracy is a reference, not an estimated oracle.
        # Only sampling uncertainty of recent correctness is represented here.
        current_se = sqrt(accuracy * (1.0 - accuracy) / observed)
        future_se = sqrt(accuracy * (1.0 - accuracy) / (observed + self.additional_samples))
        precision_value = self.consequence_weight * (current_se - future_se)
        net = precision_value - cost - delay_cost
        if deficit >= self.min_deficit and context.trials_since_update >= self.min_trials_between_updates:
            action = DecisionAction.COLLECT_MORE_EVIDENCE if net > 0 else DecisionAction.UPDATE_NOW
        else:
            action = DecisionAction.NO_UPDATE
        return EvidenceDecision(action, deficit, current_se, future_se,
                                precision_value, cost, delay_cost, net)
