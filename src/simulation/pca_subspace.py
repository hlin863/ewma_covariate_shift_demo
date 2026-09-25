"""Controlled mean shifts aligned with a training-fitted PCA basis."""
import numpy as np


SCENARIOS = ("none", "pc1", "pc2", "both")


def generate_reference(
    seed,
    n_train=500,
    n_calibration=2000,
):
    rng = np.random.default_rng(seed)
    covariance = np.array([
        [2.5, 1.5],
        [1.5, 2.5],
    ])

    training = rng.multivariate_normal(
        [0, 0], covariance, n_train
    )
    calibration = rng.multivariate_normal(
        [0, 0], covariance, n_calibration
    )

    return training, calibration


def generate_evaluation(
    seed,
    fitted_pca,
    scenario,
    magnitude=3.0,
    n_observations=1000,
    shift_at=500,
):
    if scenario not in SCENARIOS:
        raise ValueError(
            f"scenario must be one of {SCENARIOS}"
        )

    if not 0 < shift_at < n_observations or magnitude < 0:
        raise ValueError("Invalid shift position or magnitude.")

    rng = np.random.default_rng(seed)
    covariance = np.array([
        [2.5, 1.5],
        [1.5, 2.5],
    ])

    features = rng.multivariate_normal(
        [0, 0], covariance, n_observations
    )

    v1 = fitted_pca.model.components_[0]

    if v1.shape != (2,):
        raise ValueError(
            "This initial generator requires two features."
        )

    # In two dimensions this is a unit vector orthogonal to PC1.
    v2 = np.array([-v1[1], v1[0]])

    directions = {
        "none": np.zeros(2),
        "pc1": v1,
        "pc2": v2,
        "both": (v1 + v2) / np.sqrt(2),
    }

    delta = magnitude * directions[scenario]
    features[shift_at:] += delta

    true_shift = None if scenario == "none" else shift_at
    return features, true_shift, delta