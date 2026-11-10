from adaptation.classifier import build_fresh_model
from dataclasses import dataclass
import numpy as np


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
):
    """
    Cost-aware multi-fidelity adaptation optimisation.
    """

    candidates = sample_valid_configs(
        config_space,
        n=initial_candidates,
    )

    history = []

    for budget in budgets:

        scores = []

        for config in candidates:

            # Every candidate starts with the same
            # independently fitted initial model.
            model = build_fresh_model(
                development_data.x_train, development_data.y_train
            )

            # A chronological DEVELOPMENT-validation
            # prefix; never final evaluation data.
            stream, labels = make_validation_stream(
                development_data,
                budget=budget,
            )

            result = adapt_model(
                stream,
                model,
                config=config,
                evaluation_labels=labels,
            )

            cost = measure_adaptation_cost(result)

            score = (
                result.accuracy - alpha * cost.update_rate - beta * cost.normalized_time
            )

            scores.append((score, config))
            history.append((budget, config, score))

        scores.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        keep = max(
            1,
            round(len(scores) * promotion_fraction),
        )

        candidates = [config for _, config in scores[:keep]]

    return candidates[0], history
