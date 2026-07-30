from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA


@dataclass(frozen=True)
class CSEPCAResult:
    """Fitted PCA preprocessing state for covariate-shift estimation."""

    model: PCA
    training_transformed: np.ndarray
    n_input_features: int
    n_components: int
    explained_variance_ratio: np.ndarray

    @property
    def first_component(self) -> np.ndarray:
        """Return the first PCA component of the training observations."""

        return self.training_transformed[:, 0]


def _validate_feature_matrix(
    features: np.ndarray,
    *,
    minimum_observations: int = 1,
    expected_features: int | None = None,
) -> np.ndarray:
    """Convert and validate a multivariate feature matrix.

    The expected matrix shape is:

        (n_observations, n_features)

    Parameters
    ----------
    features:
        Input feature matrix.
    minimum_observations:
        Minimum permitted number of rows.
    expected_features:
        Required number of columns when applying an already fitted
        transformation.

    Returns
    -------
    np.ndarray
        Validated floating-point feature matrix.
    """

    values = np.asarray(features, dtype=float)

    if values.ndim != 2:
        raise ValueError(
            "features must be a two-dimensional matrix with shape "
            "(n_observations, n_features)."
        )

    n_observations, n_features = values.shape

    if n_observations < minimum_observations:
        raise ValueError(
            "features must contain at least "
            f"{minimum_observations} observations."
        )

    if n_features < 1:
        raise ValueError(
            "features must contain at least one feature column."
        )

    if expected_features is not None and n_features != expected_features:
        raise ValueError(
            "features contain an unexpected number of columns: "
            f"expected {expected_features}, received {n_features}."
        )

    if not np.isfinite(values).all():
        raise ValueError(
            "features must contain only finite numerical values."
        )

    return values


def _validate_n_components(
    n_components: int | float | None,
    *,
    n_observations: int,
    n_features: int,
) -> int | float | None:
    """Validate the requested PCA dimensionality."""

    maximum_components = min(n_observations, n_features)

    if n_components is None:
        return None

    if isinstance(n_components, bool):
        raise ValueError(
            "n_components must be an integer, a float in (0, 1), "
            "or None."
        )

    if isinstance(n_components, (int, np.integer)):
        component_count = int(n_components)

        if component_count < 1:
            raise ValueError(
                "integer n_components must be at least 1."
            )

        if component_count > maximum_components:
            raise ValueError(
                "integer n_components must not exceed "
                "min(n_observations, n_features), which is "
                f"{maximum_components}."
            )

        return component_count

    if isinstance(n_components, (float, np.floating)):
        variance_fraction = float(n_components)

        if not 0.0 < variance_fraction < 1.0:
            raise ValueError(
                "float n_components must be in the interval (0, 1)."
            )

        return variance_fraction

    raise ValueError(
        "n_components must be an integer, a float in (0, 1), "
        "or None."
    )


def fit_cse_pca(
    training_features: np.ndarray,
    n_components: int | float | None = None,
) -> CSEPCAResult:
    """Fit PCA on stationary CSE training features.

    PCA is fitted only on the training feature matrix. The fitted model
    must later be reused to transform testing observations so that both
    datasets remain in the same coordinate system.

    Parameters
    ----------
    training_features:
        Stationary training feature matrix with shape
        ``(n_observations, n_features)``.
    n_components:
        Number of principal components to retain. An integer retains a
        fixed number. A float in ``(0, 1)`` retains enough components to
        explain that proportion of variance. ``None`` retains every
        available component.

    Returns
    -------
    CSEPCAResult
        Fitted PCA model and transformed training features.
    """

    values = _validate_feature_matrix(
        training_features,
        minimum_observations=2,
    )

    n_observations, n_input_features = values.shape

    validated_n_components = _validate_n_components(
        n_components,
        n_observations=n_observations,
        n_features=n_input_features,
    )

    # The full solver supports integer, None and explained-variance
    # fraction forms of n_components consistently.
    model = PCA(
        n_components=validated_n_components,
        svd_solver="full",
    )

    transformed = model.fit_transform(values)

    if transformed.ndim != 2 or transformed.shape[1] < 1:
        raise RuntimeError(
            "PCA did not produce at least one principal component."
        )

    return CSEPCAResult(
        model=model,
        training_transformed=np.asarray(
            transformed,
            dtype=float,
        ),
        n_input_features=int(n_input_features),
        n_components=int(transformed.shape[1]),
        explained_variance_ratio=np.asarray(
            model.explained_variance_ratio_,
            dtype=float,
        ),
    )


def transform_cse_features(
    features: np.ndarray,
    fitted_result: CSEPCAResult,
) -> np.ndarray:
    """Transform observations using a previously fitted CSE PCA model.

    This function deliberately calls ``transform`` rather than fitting
    another PCA model. Refitting on testing data would redefine the
    feature space and invalidate comparison with the training reference.
    """

    values = _validate_feature_matrix(
        features,
        minimum_observations=1,
        expected_features=fitted_result.n_input_features,
    )

    transformed = fitted_result.model.transform(values)

    return np.asarray(
        transformed,
        dtype=float,
    )


def extract_first_component(
    transformed_features: np.ndarray,
) -> np.ndarray:
    """Extract the first PCA component as a one-dimensional signal.

    The first component is the scalar sequence monitored by the
    univariate EWMA warning stage of the CSE pipeline.
    """

    values = _validate_feature_matrix(
        transformed_features,
        minimum_observations=1,
    )

    return np.asarray(
        values[:, 0],
        dtype=float,
    )