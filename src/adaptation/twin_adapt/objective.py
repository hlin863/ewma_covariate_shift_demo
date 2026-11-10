def measure_adaptation_cost(result):
    """Measure the cost of adapting a model to a new data stream.

    Parameters
    ----------
    result : AdaptationResult
        The result of adapting a model to a new data stream.

    Returns
    -------
    AdaptationCost
        The cost of adapting the model.
    """
    updates = result.update_count
    trials = len(result.trials)

    retrain_seconds = (
        result.updates["retrain_seconds"].sum() if not result.updates.empty else 0.0
    )

    return {
        "update_rate": updates / max(1, trials),
        "retrain_seconds": retrain_seconds,
    }


from dataclasses import dataclass


@dataclass(frozen=True)
class CostObjectiveConfig:
    alpha: float = 0.2
    beta: float = 0.02
    reference_seconds: float = 1.0

    def __post_init__(self):
        if self.alpha < 0:
            raise ValueError("alpha must be nonnegative")
        if self.beta < 0:
            raise ValueError("beta must be nonnegative")
        if self.reference_seconds <= 0:
            raise ValueError("reference_seconds must be positive")
