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
