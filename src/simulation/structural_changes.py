
import numpy as np


def generate_structural_stream(
    seed: int,
    scenario: str,
    *,
    n_observations: int = 1000,
    change_at: int = 500,
    variance_ratio: float = 4.0,
    correlation_before: float = 0.0,
    correlation_after: float = 0.7,
    ar_before: float = 0.0,
    ar_after: float = 0.7,
) -> tuple[np.ndarray, dict]:
    """
    Generate a two-dimensional synthetic time-series stream.

    MATHEMATICAL OBJECTIVE
    ----------------------
    Let X_t = [X_t^(1), X_t^(2)]^T denote a bivariate observation.

    The intended structural change occurs at:

        tau = change_at

    We divide observations into two regimes:

        R_t = 0,  if t < tau       (before change)
        R_t = 1,  if t >= tau      (after change)

    The implemented time-series recursion, for t >= 1, is:

        X_t = phi_t * X_(t-1) + epsilon_t + m_t

    where:

        phi_t       = autoregressive coefficient
        epsilon_t   ~ N(0, Q_t)
        m_t         = additive mean/intercept vector
        Q_t         = Gaussian innovation covariance

    IMPORTANT:
    Q_t is the innovation covariance, not necessarily the
    stationary population covariance of X_t.

    This construction is inspired by AR-based simulations in
    Park, Im, and Lim (2023), Model B.

    It is not an exact reproduction of their experiments.
    """

    # ==============================================================
    # STEP 1 — VALIDATE THE SIMULATION CONFIGURATION
    # ==============================================================

    # n_observations = T defines the stream length.
    #
    # X = {X_0, X_1, ..., X_(T-1)}
    #
    # Each X_t contains two numerical features.
    #
    # Expected output shape:
    #
    #     X.shape = (T, 2)
    #
    # This condition permits T = 1, although one observation is
    # insufficient to study a before/after structural change.

    if n_observations <= 0:
        raise ValueError("n_observations must be positive.")

    # change_at = tau identifies the change-point index.
    #
    # For the default configuration:
    #
    #     tau = 500
    #     T   = 1000
    #
    # Before: t = 0, ..., 499
    # After:  t = 500, ..., 999
    #
    # Boundary cases:
    #
    #     tau = 0  : no pre-change segment
    #     tau = T  : no post-change segment
    #
    # Both are accepted by the current code.

    if not 0 <= change_at <= n_observations:
        raise ValueError("change_at must be between 0 and n_observations.")

    # A variance scaling ratio must be positive.
    #
    #     variance_ratio = r_v > 0
    #
    # For r_v = 4, the covariance matrix receives
    # diagonal elements of 4 in the post-change regime.
    #
    # NOTE:
    # This does not guarantee the observed process has
    # a population variance ratio of exactly 4 when phi changes.

    if variance_ratio <= 0:
        raise ValueError("variance_ratio must be positive.")

    # ==============================================================
    # STEP 2 — INITIALISE A REPRODUCIBLE RANDOM GENERATOR
    # ==============================================================

    # The seed determines the pseudorandom sequence.
    #
    # Given the same seed and the same configuration,
    # the simulation should generate the same observations.
    #
    # This is essential when comparing change detectors
    # on identical synthetic streams.

    rng = np.random.default_rng(seed)

    # Normalise the scenario identifier to lowercase.
    #
    # Example:
    #
    #     "MEAN_SHIFT" -> "mean_shift"
    #
    # NOTE:
    # The code does not remove leading/trailing whitespace
    # and does not reject unknown scenario names.

    scenario_key = scenario.lower()

    # ==============================================================
    # STEP 3 — DEFINE CORRELATION AND AR PARAMETERS
    # ==============================================================

    # Correlation measures linear dependence between dimensions:
    #
    #     rho = Cov(X1, X2) / (sigma1 * sigma2)
    #
    # The theoretical correlation range is:
    #
    #     -1 <= rho <= 1
    #
    # Here values are clipped to [-0.99, 0.99].
    #
    # Example:
    #     correlation_after = 1.2 becomes 0.99.
    #
    # The clipping avoids extreme parameter values but silently
    # modifies the requested configuration.

    corr_before = float(np.clip(correlation_before, -0.99, 0.99))
    corr_after = float(np.clip(correlation_after, -0.99, 0.99))

    # phi determines temporal dependence:
    #
    #     X_t = phi * X_(t-1) + epsilon_t + m
    #
    # Interpretation:
    #
    #     phi = 0   -> no AR(1) dependence
    #     phi = 0.7 -> persistent positive dependence
    #     phi < 0   -> alternating/sign-reversing tendency
    #
    # An AR(1) process is stationary when:
    #
    #     |phi| < 1
    #
    # The current clipping restricts phi to [-0.99, 0.99].

    ar_prev = float(np.clip(ar_before, -0.99, 0.99))
    ar_post = float(np.clip(ar_after, -0.99, 0.99))

    # ==============================================================
    # STEP 4 — CONSTRUCT THE PRE-CHANGE COVARIANCE MATRIX
    # ==============================================================

    # Baseline innovation covariance:
    #
    #              [ 1       rho_0 ]
    #     Q_0 =    [               ]
    #              [ rho_0   1     ]
    #
    # This specifies:
    #
    #     Var(epsilon_1) = 1
    #     Var(epsilon_2) = 1
    #     Cov(epsilon_1, epsilon_2) = rho_0
    #
    # With correlation_before = 0:
    #
    #              [ 1  0 ]
    #     Q_0 =    [      ]
    #              [ 0  1 ]
    #
    # This is an identity covariance matrix.

    cov_before = np.array(
        [[1.0, corr_before], [corr_before, 1.0]],
        dtype=float
    )

    # ==============================================================
    # STEP 5 — CONSTRUCT THE POST-CHANGE COVARIANCE MATRIX
    # ==============================================================

    # Your implementation defines:
    #
    #              [ r_v             rho_1 * sqrt(r_v) ]
    #     Q_1 =    [                                   ]
    #              [ rho_1 * sqrt(r_v)       r_v        ]
    #
    # For the defaults:
    #
    #     r_v   = 4
    #     rho_1 = 0.7
    #
    #              [ 4    1.4 ]
    #     Q_1 =    [          ]
    #              [ 1.4  4   ]
    #
    # IMPORTANT MATHEMATICAL DISTINCTION:
    #
    # The actual innovation correlation is:
    #
    #     Corr = 1.4 / sqrt(4 * 4)
    #          = 1.4 / 4
    #          = 0.35
    #
    # Therefore the specified rho_1 = 0.7 is not achieved.
    #
    # For equal marginal variances r_v, the correct covariance
    # for a desired correlation rho_1 would be:
    #
    #              [ r_v         rho_1 * r_v ]
    #     Q_1 =    [                         ]
    #              [ rho_1 * r_v      r_v    ]
    #
    # We preserve your original calculation below for review.

    cov_after = np.array(
        [
            [variance_ratio, corr_after * np.sqrt(variance_ratio)],
            [corr_after * np.sqrt(variance_ratio), variance_ratio],
        ],
        dtype=float,
    )

    # ==============================================================
    # STEP 6 — DEFINE PRE-CHANGE AND POST-CHANGE MEAN INPUTS
    # ==============================================================

    # Baseline additive mean vector:
    #
    #             [ 0 ]
    #     m_0 =   [   ]
    #             [ 0 ]
    #
    # There is no additive offset before the structural change.

    mean_before = np.zeros(2, dtype=float)

    # Post-change additive mean vector:
    #
    #             [ 2 ]
    #     m_1 =   [   ]
    #             [ 1 ]
    #
    # INTERPRETATION:
    #
    # The first component receives an additive offset of 2,
    # and the second component an additive offset of 1.
    #
    # However, for the recursion
    #
    #     X_t = phi * X_(t-1) + epsilon_t + m
    #
    # the stationary population mean is:
    #
    #     E[X] = m / (1 - phi)
    #
    # With phi = 0.7, the post-change stationary mean is:
    #
    #     E[X] = [6.667, 3.333]
    #
    # rather than [2, 1].

    mean_after = np.array([2.0, 1.0], dtype=float)

    # ==============================================================
    # STEP 7 — MARK THE PRE/POST-CHANGE REGIMES
    # ==============================================================

    # Create a Boolean vector:
    #
    #     switched[t] = False, t < tau
    #     switched[t] = True,  t >= tau
    #
    # For T = 1000 and tau = 500:
    #
    #     [False, ..., False, True, ..., True]
    #        first 500         last 500
    #
    # NOTE:
    # In this implementation, switched is constructed but
    # is not subsequently used or returned.

    switched = np.zeros(n_observations, dtype=bool)
    switched[change_at:] = True

    # ==============================================================
    # STEP 8 — ALLOCATE THE STREAM
    # ==============================================================

    # X is a matrix in R^(T x 2):
    #
    #             [ X_0^(1)     X_0^(2)   ]
    #             [ X_1^(1)     X_1^(2)   ]
    #     X =     [    ...         ...    ]
    #             [ X_(T-1)^(1) X_(T-1)^(2)]
    #
    # Each row is an observation at time i.
    # Each column represents one feature/dimension.

    stream = np.zeros((n_observations, 2), dtype=float)

    # ==============================================================
    # STEP 9 — GENERATE OBSERVATIONS SEQUENTIALLY
    # ==============================================================

    for i in range(n_observations):

        # ----------------------------------------------------------
        # 9.1 — INITIAL PARAMETER SELECTION
        # ----------------------------------------------------------

        # At i = 0, initialise from baseline parameters.
        #
        # NOTE:
        # Even when change_at = 0, this branch uses
        # pre-change parameters for the first observation.

        if i == 0:
            mean = mean_before
            cov = cov_before

        else:
            # Select the additive mean and innovation covariance
            # based on the change-point index:
            #
            #    m_i = m_0,  i < tau
            #    m_i = m_1,  i >= tau
            #
            #    Q_i = Q_0,  i < tau
            #    Q_i = Q_1,  i >= tau
            #
            # These are initial choices that the scenario-specific
            # branches below may override.

            mean = mean_after if i >= change_at else mean_before
            cov = cov_after if i >= change_at else cov_before

        # ----------------------------------------------------------
        # 9.2 — SELECT THE SCENARIO
        # ----------------------------------------------------------

        if scenario_key == "mean_shift":

            # Intended experiment:
            #
            #     m_0 != m_1
            #
            # A mean change is introduced at tau.
            #
            # IMPORTANT:
            # This branch does not reset cov or phi.
            # With default parameters, covariance and
            # AR dependence change as well.

            mean = mean_before if i < change_at else mean_after

        elif scenario_key == "variance_shift":

            # Intended experiment:
            #
            #     Var_before != Var_after
            #
            # The additive mean is fixed at zero.
            #
            # IMPORTANT:
            # Covariance and AR coefficients still change
            # under the initial regime-selection logic.
            #
            # Thus this is not necessarily variance-only.

            mean = mean_before

        elif scenario_key in {"correlation_shift", "covariance_shift"}:

            # Intended experiment:
            #
            #     Corr_before != Corr_after
            #
            # Fix the additive mean at zero.
            #
            # Explicitly select Q_0 or Q_1.
            #
            # However, Q_1 also changes diagonal variance
            # through variance_ratio.
            #
            # The AR coefficient also changes by default.

            mean = mean_before
            cov = cov_before if i < change_at else cov_after

        elif scenario_key in {"ar_shift", "autocorrelation_shift"}:

            # Intended experiment:
            #
            #     phi_before != phi_after
            #
            # A change in phi modifies temporal dependence:
            #
            #     Corr(X_t, X_(t-1)) = phi
            #
            # for each stationary AR(1) component.
            #
            # The mean input is fixed to zero.
            #
            # IMPORTANT:
            # The covariance still changes by default.
            #
            # Therefore this is not an isolated AR shift.

            mean = mean_before

        elif scenario_key == "combined":

            # Intended experiment:
            #
            #     mean change
            #       +
            #     covariance change
            #       +
            #     AR coefficient change
            #
            # This branch explicitly selects the regime
            # mean and covariance.
            #
            # With default settings, it changes all three.

            mean = mean_before if i < change_at else mean_after
            cov = cov_before if i < change_at else cov_after

        else:

            # Fallback for unknown scenario names.
            #
            # Current behaviour:
            #     generate the same general combined regime
            #     instead of raising a configuration error.
            #
            # This can conceal spelling mistakes in experiments.

            mean = mean_before if i < change_at else mean_after
            cov = cov_before if i < change_at else cov_after

        # ----------------------------------------------------------
        # 9.3 — INITIAL OBSERVATION
        # ----------------------------------------------------------

        if i == 0:

            # Initial observation:
            #
            #     X_0 ~ N(m_0, Q_0)
            #
            # The generator samples one bivariate Gaussian vector.
            #
            # IMPORTANT:
            # When phi_before != 0, Q_0 is not the stationary
            # population covariance of the AR process.

            stream[i] = rng.multivariate_normal(mean, cov)

        # ----------------------------------------------------------
        # 9.4 — AUTOREGRESSIVE STREAM UPDATE
        # ----------------------------------------------------------

        else:

            # Select the regime-specific AR coefficient:
            #
            #     phi_i = phi_before, i < tau
            #     phi_i = phi_after,  i >= tau
            #
            # NOTE:
            # This AR switch is applied across all scenarios,
            # not only "ar_shift".

            phi = ar_prev if i < change_at else ar_post

            # Generate the next observation:
            #
            #     X_i = phi_i * X_(i-1) + epsilon_i + m_i
            #
            # where:
            #
            #     epsilon_i ~ N(0, Q_i)
            #
            # phi_i * X_(i-1):
            #     temporal memory from the previous observation.
            #
            # epsilon_i:
            #     new bivariate Gaussian innovation.
            #
            # m_i:
            #     additive mean/intercept input.
            #
            # This AR(1) formula is the central mathematical
            # structure of your synthetic stream.
            #
            # It allows first-order temporal dependence and
            # potentially changing second-order properties.

            stream[i] = (
                phi * stream[i - 1]
                + rng.multivariate_normal(np.zeros(2), cov)
                + mean
            )

    # ==============================================================
    # STEP 10 — CONSTRUCT EXPERIMENT METADATA
    # ==============================================================

    # The metadata records configuration settings:
    #
    #     T       = total observations
    #     tau     = change-point index
    #     rho     = supplied correlation parameter
    #     phi     = supplied AR parameter
    #     r_v     = supplied variance-scaling parameter
    #
    # LIMITATION:
    # These values are not necessarily the actual
    # stationary population moments of the generated stream.
    #
    # In particular, the metadata does not directly record:
    #
    #     E[X] before / after
    #     Var(X) before / after
    #     Cov(X1, X2) before / after
    #     Cov(X_t, X_(t-1)) before / after

    metadata = {
        "scenario": scenario,
        "n_observations": n_observations,
        "change_at": change_at,
        "variance_ratio": variance_ratio,
        "correlation_before": corr_before,
        "correlation_after": corr_after,
        "ar_before": ar_prev,
        "ar_after": ar_post,
        "shape": stream.shape,
    }

    # Return:
    #
    #     stream   : ndarray with shape (T, 2)
    #     metadata : dictionary of simulation parameters
    #
    # The returned observations can be supplied to a
    # structural-change detector for evaluation.

    return stream, metadata
