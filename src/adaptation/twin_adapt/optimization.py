from dataclasses import dataclass
from time import perf_counter
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
        if self.y_train.ndim != 1 or self.y_validation.ndim != 1:
            raise ValueError("Training and validation labels must be one-dimensional")
        if len(self.x_train) != len(self.y_train):
            raise ValueError("x_train and y_train must have the same length")
        if len(self.x_train) == 0:
            raise ValueError("Training data must not be empty")
        if np.unique(self.y_train).size < 2:
            raise ValueError("SVM training requires at least two training classes")
        if (
            not np.isfinite(self.x_train).all()
            or not np.isfinite(self.x_validation).all()
        ):
            raise ValueError("All features must be finite")
        if len(self.x_validation) != len(self.y_validation):
            raise ValueError("x_validation and y_validation must have the same length.")
        if self.x_train.shape[1] != self.x_validation.shape[1]:
            raise ValueError(
                "x_train and x_validation must have the same number of features."
            )
        if len(self.x_validation) == 0:
            raise ValueError("x_validation and y_validation must not be empty.")


def compare_candidate_set(candidate_results, atol=1e-8):
    """Compare candidate predictions and objective scores."""

    if not candidate_results:
        raise ValueError("No candidate results to compare")

    scores = np.array(
        [item["score"] for item in candidate_results],
        dtype=float,
    )

    predictions = [
        item["result"].trials["prediction"].to_numpy() for item in candidate_results
    ]

    trial_indices = [
        item["result"].trials["trial_index"].to_numpy() for item in candidate_results
    ]

    # Ensure all candidates used the same validation trials
    for indices in trial_indices[1:]:
        if not np.array_equal(indices, trial_indices[0]):
            raise ValueError("Candidate validation trials differ")

    disagreements = []

    for i in range(len(predictions)):
        for j in range(i + 1, len(predictions)):
            difference = np.mean(predictions[i] != predictions[j])
            disagreements.append(float(difference))

    prediction_diversity = max(disagreements) if disagreements else 0.0

    score_spread = float(np.max(scores) - np.min(scores))

    scores_equal = bool(np.allclose(scores, scores[0], atol=atol, rtol=0.0))

    predictions_equal = prediction_diversity == 0.0

    if scores_equal and predictions_equal:
        status = "prediction_and_score_tied"
    elif scores_equal:
        status = "score_tied"
    else:
        status = "ranked"

    return {
        "selection_status": status,
        "prediction_diversity": prediction_diversity,
        "score_spread": score_spread,
        "scores_equal": scores_equal,
        "predictions_equal": predictions_equal,
    }


def optimize_adaptation(
    development_data: DevelopmentData,
    config_space: TwinAdaptSearchSpace,
    budgets,
    initial_candidates: int = 12,
    promotion_fraction: float = 1 / 3,
    *,
    objective: CostObjectiveConfig | None = None,
) -> tuple[TwinAdaptConfig, list[dict]]:
    """Search training-only configurations using progressively longer validation prefixes.

    Return the best candidate at the largest budget and per-evaluation
    provenance. Validation labels are released only through the existing controller.
    """
    if objective is None:
        objective = CostObjectiveConfig()
    budgets = list(budgets)
    if not budgets or any(type(b) is not int or b <= 0 for b in budgets):
        raise ValueError("budgets must be nonempty positive integers")
    if budgets != sorted(set(budgets)):
        raise ValueError("budgets must be strictly increasing")
    if budgets[-1] > len(development_data.x_validation):
        raise ValueError("largest budget exceeds development validation trials")
    if budgets[0] < config_space.batch_size:
        raise ValueError("smallest budget must contain a complete batch")
    if not 0 < promotion_fraction <= 1:
        raise ValueError("promotion_fraction must be in (0, 1]")

    candidates = sample_valid_configs(config_space, n=initial_candidates)
    history: list[dict] = []
    best_config: TwinAdaptConfig | None = None

    for stage, budget in enumerate(budgets):
        scores = []

        candidate_results = []

        for candidate in candidates:
            model = build_fresh_model(
                development_data.x_train, development_data.y_train, seed=candidate.seed
            )
            stream, labels = make_validation_stream(
                development_data.x_validation,
                development_data.y_validation,
                batch_size=candidate.batch_size,
                budget=budget,
            )
            started = perf_counter()

            result = adapt_model(
                stream,
                model,
                config=candidate,
                evaluation_labels=labels,
            )

            elapsed = perf_counter() - started
            cost = measure_adaptation_cost(result)

            score = calculate_objective_score(
                accuracy=result.accuracy,
                cost=cost,
                objective=objective,
            )

            if not np.isfinite(score):
                raise ValueError("candidate objective score must be finite")

            record = {
                "stage": stage,
                "budget": budget,
                "candidate": candidate.snapshot(),
                "accuracy": float(result.accuracy),
                "updates": int(result.update_count),
                "update_rate": float(cost["update_rate"]),
                "retrain_seconds": float(cost["retrain_seconds"]),
                "replay_seconds": float(elapsed),
                "score": float(score),
                "promoted": False,
            }

            history.append(record)
            scores.append((score, candidate, record))

            candidate_results.append(
                {
                    "candidate": candidate,
                    "result": result,
                    "score": float(score),
                    "record": record,
                }
            )

        # Stable sorting preserves the seeded candidate order on exact score ties.
        scores.sort(key=lambda entry: entry[0], reverse=True)

        # NEW: compare candidates evaluated on the same budget
        diagnostics = compare_candidate_set(candidate_results)

        for comparison in candidate_results:
            record = comparison["record"]

            record["selection_status"] = diagnostics["selection_status"]
            record["prediction_diversity"] = diagnostics["prediction_diversity"]
            record["score_spread"] = diagnostics["score_spread"]

        best_config = scores[0][1]
        if stage < len(budgets) - 1:
            keep = max(1, int(np.ceil(len(scores) * promotion_fraction)))
        else:
            keep = 1
        for _, _, record in scores[:keep]:
            record["promoted"] = True
        candidates = [candidate for _, candidate, _ in scores[:keep]]

    assert best_config is not None
    return best_config, history
