from dataclasses import dataclass
import numpy as np
from itertools import product

from src.adaptation.classifier import build_fresh_model
from src.adaptation.twin_adapt.config import TwinAdaptConfig
from src.adaptation.twin_adapt.objective import (
    measure_adaptation_cost,
    CostObjectiveConfig,
)
from src.adaptation.twin_adapt.controller import adapt_model
from src.online.batch_stream import make_validation_stream


def calculate_objective_score(
    accuracy: float, cost: dict, objective: CostObjectiveConfig
) -> float:
    """
    Calculate the objective score for a given accuracy and cost.

    Parameters
    ----------
    accuracy : float
        The accuracy of the model.
    cost : dict
        The cost of adapting the model, containing 'update_rate' and 'retrain_seconds'.
    objective : CostObjectiveConfig
        The configuration for the cost-aware objective.

    Returns
    -------
    float
        The calculated objective score.
    """
    update_rate = cost.get("update_rate", 0.0)
    retrain_seconds = cost.get("retrain_seconds", 0.0)

    normalized_time = retrain_seconds / objective.reference_seconds

    score = (
        accuracy - (objective.alpha * update_rate) - (objective.beta * normalized_time)
    )

    return score


@dataclass(frozen=True)
class TwinAdaptSearchSpace:
    """Candidate values for TWIN-ADAPT optimisation."""

    gamma_values: tuple[float, ...] = (0.93, 0.95, 0.97)

    delta_values: tuple[float, ...] = (0.80, 0.85, 0.90)

    win_size_values: tuple[int, ...] = (5, 10, 15)

    win_size_max_values: tuple[int, ...] = (80, 120, 160)

    batch_size: int = 40
    label_delay: int | None = 0
    seed: int = 42


def sample_valid_configs(
    config_space: TwinAdaptSearchSpace,
    *,
    n: int = 12,
) -> list[TwinAdaptConfig]:
    """
    Sample unique valid TWIN-ADAPT configurations.

    Returns configurations in a reproducibly
    shuffled order.
    """

    if n <= 0:
        raise ValueError("n must be a positive integer")

    candidates = []

    # Construct the Cartesian product of
    # candidate hyperparameter values.
    combinations = product(
        config_space.gamma_values,
        config_space.delta_values,
        config_space.win_size_values,
        config_space.win_size_max_values,
    )

    for gamma, delta, win_size, win_size_max in combinations:

        try:
            config = TwinAdaptConfig(
                batch_size=config_space.batch_size,
                win_size=win_size,
                win_size_max=win_size_max,
                gamma=gamma,
                delta=delta,
                label_delay=config_space.label_delay,
                seed=config_space.seed,
            )

        except ValueError:
            # Ignore invalid configurations,
            # e.g. delta >= gamma.
            continue

        candidates.append(config)

    # Ensure configurations are unique.
    candidates = list(dict.fromkeys(candidates))

    if len(candidates) < n:
        raise ValueError(
            f"Requested {n} candidates, but only "
            f"{len(candidates)} valid configurations exist."
        )

    # Reproducible sampling without replacement.
    rng = np.random.default_rng(config_space.seed)

    indices = rng.choice(
        len(candidates),
        size=n,
        replace=False,
    )

    return [candidates[i] for i in indices]


@dataclass(frozen=True)
class DevelopmentData:
    x_train: np.ndarray
    y_train: np.ndarray

    x_validation: np.ndarray
    y_validation: np.ndarray

    def __post_init__(self):
        if self.x_train.ndim != 2:
            raise ValueError("x_train must be a 2D array.")
        if self.x_validation.ndim != 2:
            raise ValueError("x_validation must be a 2D array.")
        if len(self.x_validation) != len(self.y_validation):
            raise ValueError("x_validation and y_validation must have the same length.")
        if self.x_train.shape[1] != self.x_validation.shape[1]:
            raise ValueError(
                "x_train and x_validation must have the same number of features."
            )
        if len(self.x_validation) == 0:
            raise ValueError("x_validation and y_validation must not be empty.")


def optimize_adaptation(
    development_data,
    config_space,
    budgets,
    initial_candidates=12,
    promotion_fraction=1 / 3,
    *,
    objective: CostObjectiveConfig | None = None,
) -> tuple[TwinAdaptConfig, list[tuple[int, TwinAdaptConfig, float]]]:
    """
    Cost-aware multi-fidelity adaptation optimisation.
    """
    if objective is None:
        objective = CostObjectiveConfig()

    candidates = sample_valid_configs(
        config_space,
        n=initial_candidates,
    )

    history = []

    for budget in budgets:

        scores = []

        for candidate in candidates:

            # Every candidate starts with the same
            # independently fitted initial model.
            model = build_fresh_model(
                development_data.x_train, development_data.y_train, seed=candidate.seed
            )

            # A chronological DEVELOPMENT-validation
            # prefix; never final evaluation data.
            stream, labels = make_validation_stream(
                development_data.x_validation,
                development_data.y_validation,
                batch_size=candidate.batch_size,
                budget=budget,
            )

            result = adapt_model(
                stream,
                model,
                config=candidate,
                evaluation_labels=labels,
            )

            cost = measure_adaptation_cost(result)

            score = calculate_objective_score(
                accuracy=result.accuracy, cost=cost, objective=objective
            )

            scores.append((score, candidate))
            history.append((budget, candidate, score))

        scores.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        keep = max(
            1,
            round(len(scores) * promotion_fraction),
        )

        candidates = [candidate for _, candidate in scores[:keep]]

    return candidates[0], history
