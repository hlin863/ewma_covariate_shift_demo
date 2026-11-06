from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EWMATrainingResult:
    lambda_value: float
    initial_z: float
    final_z: float
    error_variance: float
    error_standard_deviation: float


def _validate_univariate_values(
    values: np.ndarray,
    *,
    minimum_size: int = 1,
) -> np.ndarray:
    observations = np.asarray(values, dtype=float)
    if observations.ndim != 1:
        raise ValueError("values must be one-dimensional.")
    if observations.size < minimum_size:
        raise ValueError(f"values must contain at least {minimum_size} observations.")
    if not np.isfinite(observations).all():
        raise ValueError("values must contain only finite observations.")
    return observations


def calculate_ewma_training_path(
    values: np.ndarray,
    lambda_value: float,
    initial_z: float,
) -> tuple[np.ndarray, np.ndarray]:
    observations = np.asarray(values, dtype=float)
    if observations.ndim != 1:
        raise ValueError("values must be one-dimensional.")
    if not 0.0 < lambda_value <= 1.0:
        raise ValueError("lambda_value must be in (0, 1].")

    ewma_values = np.empty(observations.size)
    prediction_errors = np.empty(observations.size)
    previous_z = float(initial_z)
    for index, observation in enumerate(observations):
        prediction_errors[index] = observation - previous_z
        current_z = lambda_value * observation + (1.0 - lambda_value) * previous_z
        ewma_values[index] = current_z
        previous_z = current_z
    return ewma_values, prediction_errors


def estimate_lambda(
    values: np.ndarray,
    candidates: np.ndarray | None = None,
) -> tuple[float, pd.DataFrame]:
    observations = np.asarray(values, dtype=float)
    if candidates is None:
        candidates = np.arange(0.01, 1.01, 0.01)
    initial_z = float(observations.mean())
    records: list[dict[str, float]] = []
    for candidate in candidates:
        _, errors = calculate_ewma_training_path(
            values=observations,
            lambda_value=float(candidate),
            initial_z=initial_z,
        )
        records.append({"lambda": float(candidate), "sse": float(np.sum(errors**2))})
    search_results = pd.DataFrame(records)
    best_index = search_results["sse"].idxmin()
    best_lambda = float(search_results.loc[best_index, "lambda"])
    return best_lambda, search_results


def fit_sd_ewma(
    training_values: np.ndarray,
    *,
    lambda_override: float | None = None,
) -> EWMATrainingResult:
    observations = _validate_univariate_values(training_values, minimum_size=2)
    if lambda_override is not None:
        if not 0.0 < lambda_override <= 1.0:
            raise ValueError("lambda_override must be in the interval (0, 1].")
        effective_lambda = float(lambda_override)
    else:
        effective_lambda, _ = estimate_lambda(observations)

    initial_z = float(observations.mean())
    ewma_values, errors = calculate_ewma_training_path(
        values=observations,
        lambda_value=effective_lambda,
        initial_z=initial_z,
    )
    error_variance = float(np.mean(errors**2))
    return EWMATrainingResult(
        lambda_value=effective_lambda,
        initial_z=initial_z,
        final_z=float(ewma_values[-1]),
        error_variance=error_variance,
        error_standard_deviation=float(np.sqrt(error_variance)),
    )


@dataclass(frozen=True)
class SD_EWMA_Config:
    lambda_value: float
    variance_smoothing: float = 0.05
    control_limit_multiplier: float = 1.96
    # Derived from the paper's alpha = 0.05 under its normal error assumptions even though L itself is not separately reported from the experimental configuration.
    variance_update_mode: str = "always"

    def __post_init__(self) -> None:
        if not 0.0 < self.lambda_value <= 1.0:
            raise ValueError("lambda_value must be in (0, 1].")
        if not 0.0 < self.variance_smoothing <= 1.0:
            raise ValueError("variance_smoothing must be in (0, 1].")
        if self.control_limit_multiplier <= 0.0:
            raise ValueError("control_limit_multiplier must be positive.")
        if self.variance_update_mode not in {"always", "frozen", "non_alarm"}:
            raise ValueError(
                "variance_update_mode must be 'always', 'frozen', or 'non_alarm'."
            )


def run_sd_ewma(
    values: np.ndarray,
    times: np.ndarray,
    initial_z: float,
    initial_error_variance: float,
    config: SD_EWMA_Config,
) -> pd.DataFrame:
    observations = _validate_univariate_values(values)
    time_values = np.asarray(times, dtype=float)

    if time_values.ndim != 1:
        raise ValueError("times must be one-dimensional.")
    if observations.size != time_values.size:
        raise ValueError("values and times must have equal length.")
    if not np.isfinite(time_values).all():
        raise ValueError("times must be finite.")
    if not np.equal(time_values, np.floor(time_values)).all():
        raise ValueError("times must contain integer observation indices.")
    if np.any(np.diff(time_values) <= 0):
        raise ValueError("times must be strictly increasing.")
    if initial_error_variance <= 0.0 or not np.isfinite(initial_error_variance):
        raise ValueError("initial_error_variance must be positive and finite.")
    if not np.isfinite(initial_z):
        raise ValueError("initial_z must be finite.")

    state = SD_EWMA_State(
        z=float(initial_z),
        error_variance=float(initial_error_variance),
    )
    records: list[dict[str, float | int | str]] = []
    for time, observation in zip(time_values, observations):
        state, record = step_sd_ewma(
            state, observation=float(observation), time=int(time), config=config
        )
        records.append(record)
    return pd.DataFrame(records)


@dataclass(frozen=True)
class SD_EWMA_State:
    """
    State for SD-EWMA.

    Attributes:
        z: Current EWMA value.
        error_variance: Current error variance.
        last_time: Last observation time (default: None).
        n_observations: Number of observations processed (default: 0).
    """

    z: float
    error_variance: float
    last_time: int | None = None
    n_observations: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "z", float(self.z))
        object.__setattr__(self, "error_variance", float(self.error_variance))
        if not np.isfinite(self.z):
            raise ValueError("z must be finite")
        if not np.isfinite(self.error_variance) or self.error_variance <= 0:
            raise ValueError("error_variance must be positive and finite")

        if self.last_time is not None:
            if isinstance(self.last_time, (bool, np.bool_)) or not isinstance(
                self.last_time, (int, np.integer)
            ):
                raise ValueError("last_time must be an integer or None")
            object.__setattr__(self, "last_time", int(self.last_time))

        if (
            isinstance(self.n_observations, (bool, np.bool_))
            or not isinstance(self.n_observations, (int, np.integer))
            or self.n_observations < 0
        ):
            raise ValueError("n_observations must be a non-negative integer")
        object.__setattr__(self, "n_observations", int(self.n_observations))


def step_sd_ewma(
    state: SD_EWMA_State,
    observation: float,
    time: int,
    config: SD_EWMA_Config,
) -> tuple[SD_EWMA_State, dict[str, float | int | str]]:
    """Process one scalar observation; preserve legacy batch semantics and trace schema."""
    x = float(observation)
    if not np.isfinite(x):
        raise ValueError("observation must be finite.")
    if isinstance(time, (bool, np.bool_)) or not isinstance(time, (int, np.integer)):
        raise ValueError("time must be an integer observation index.")
    if state.last_time is not None and int(time) <= state.last_time:
        raise ValueError("time must be strictly increasing.")

    previous_z = state.z
    previous_variance = state.error_variance
    previous_std = float(np.sqrt(previous_variance))
    half_width = config.control_limit_multiplier * previous_std
    lcl = previous_z - half_width
    ucl = previous_z + half_width
    prediction_error = float(x - previous_z)
    stage_1_alarm = bool(x <= lcl or x >= ucl)
    current_z = (
        config.lambda_value * x + (1.0 - config.lambda_value) * previous_z
    )
    updated_variance = (
        config.variance_smoothing * prediction_error**2
        + (1.0 - config.variance_smoothing) * previous_variance
    )
    variance_updated = config.variance_update_mode == "always" or (
        config.variance_update_mode == "non_alarm" and not stage_1_alarm
    )
    current_variance = (
        float(updated_variance) if variance_updated else previous_variance
    )
    next_state = SD_EWMA_State(
        z=float(current_z),
        error_variance=float(current_variance),
        last_time=int(time),
        n_observations=state.n_observations + 1,
    )
    record: dict[str, float | int | str] = {
        "time": int(time),
        "x": float(x),
        "prediction": float(previous_z),
        "ewma": float(current_z),
        "error": prediction_error,
        "error_variance_before": float(previous_variance),
        "error_std_before": previous_std,
        "control_limit_multiplier": float(config.control_limit_multiplier),
        "control_limit_half_width": float(half_width),
        "lcl": float(lcl),
        "ucl": float(ucl),
        "stage_1_alarm": int(stage_1_alarm),
        "error_variance_after": float(current_variance),
        "variance_update_mode": config.variance_update_mode,
        "lambda": float(config.lambda_value),
        "variance_smoothing": float(config.variance_smoothing),
        "variance_updated": int(variance_updated),
    }
    return next_state, record


class OnlineSDEWMA:
    """Stateful, one-scalar-at-a-time Stage-I detector."""

    def __init__(self, state: SD_EWMA_State, config: SD_EWMA_Config) -> None:
        self.state = state
        self.config = config

    @classmethod
    def from_training(
        cls, training: EWMATrainingResult, config: SD_EWMA_Config
    ) -> "OnlineSDEWMA":
        if not np.isclose(training.lambda_value, config.lambda_value, rtol=0, atol=1e-12):
            raise ValueError("config lambda_value must match training lambda_value.")
        return cls(
            SD_EWMA_State(
                z=training.final_z, error_variance=training.error_variance
            ),
            config,
        )

    def update_one(self, observation: float, time: int) -> dict[str, float | int | str]:
        next_state, record = step_sd_ewma(
            self.state, observation=observation, time=time, config=self.config
        )
        # The state only advances if the entire transition succeeds.
        self.state = next_state
        return record

    def update(self, observation: float, time: int) -> dict[str, float | int | str]:
        """Alias for the one-observation API."""
        return self.update_one(observation, time)
