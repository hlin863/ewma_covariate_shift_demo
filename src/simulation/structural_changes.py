
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
    """Generate a two-feature piecewise AR(1) stream with a known breakpoint.

    Model (t >= 1):
        X_t = mu_r + phi_r * (X_(t-1) - mu_r) + epsilon_t
        epsilon_t ~ N(0, Q_r),  r = 1[t >= change_at]

    mu_r is the target stationary mean, phi_r the AR coefficient and Q_r
    the innovation covariance. For |phi_r| < 1, the stationary process
    covariance is Gamma_r = Q_r/(1 - phi_r**2), NOT Q_r. At a regime
    transition, observations initially retain previous-regime dependence.

    Park, Im & Lim (2023), Section 3.1: Model A changes AR-series
    means; Model B(i) changes AR coefficients; Model B(ii) introduces
    correlated innovations; Model C changes MA coefficients. This
    two-feature generator is Park-inspired, NOT an exact reproduction of
    the paper's 50-dimensional simulations or wavelet/DPCA detector.

    Scenario labels express intent: in this version covariance and AR
    coefficients can change together across multiple named scenarios.
    Returned metadata records parameters, not exact population moments.
    """

    # T = n_observations; tau = change_at (first post-change index).
    # tau=0 and tau=T are accepted boundary cases, not two-regime streams.
    if n_observations <= 0:
        raise ValueError("n_observations must be positive.")

    if not 0 <= change_at <= n_observations:
        raise ValueError("change_at must be between 0 and n_observations.")

    if variance_ratio <= 0:
        raise ValueError("variance_ratio must be positive.")

    # Local seeded generator ensures reproducibility for fixed parameters.
    rng = np.random.default_rng(seed)

    scenario_key = scenario.lower()

    # rho is the innovation cross-correlation. Clipping silently adjusts
    # invalid requested values; the function does not reject them.
    corr_before = float(np.clip(correlation_before, -0.99, 0.99))
    corr_after = float(np.clip(correlation_after, -0.99, 0.99))

    # phi controls serial dependence; |phi| < 1 for stable AR(1) regimes.
    ar_prev = float(np.clip(ar_before, -0.99, 0.99))
    ar_post = float(np.clip(ar_after, -0.99, 0.99))

    # Q_0 = [[1, rho_0], [rho_0, 1]]: pre-change innovation covariance.
    cov_before = np.array(
        [[1.0, corr_before], [corr_before, 1.0]],
        dtype=float
    )

    # Q_1 = v*[[1, rho_1], [rho_1, 1]], v = variance_ratio.
    # Defaults: Q_1=[[4,2.8],[2.8,4]], innovation correlation = 0.7.
    # v is NOT necessarily the stationary process variance ratio when phi changes.
    cov_after = np.array(
        [
            [variance_ratio, corr_after * variance_ratio],
            [corr_after * variance_ratio, variance_ratio],
        ],
        dtype=float,
    )

    # mu_0 and mu_1 are target stationary means, NOT AR intercepts:
    # intercept c_r = (1 - phi_r)*mu_r in the mean-centred recursion.
    mean_before = np.zeros(2, dtype=float)

    mean_after = np.array([2.0, 1.0], dtype=float)

    # Mark post-break observations; switched is not consumed or returned yet.
    switched = np.zeros(n_observations, dtype=bool)
    switched[change_at:] = True

    # Stream rows are two-dimensional observations X_t; shape = (T, 2).
    stream = np.zeros((n_observations, 2), dtype=float)

    for i in range(n_observations):

        # Choose default regime parameters, then apply scenario overrides.
        # At tau=0 the first observation still uses baseline parameters.
        if i == 0:
            mean = mean_before
            cov = cov_before

        else:

            mean = mean_after if i >= change_at else mean_before
            cov = cov_after if i >= change_at else cov_before

        # Scenarios are not yet isolated: covariance and phi change by default.
        # variance/correlation/AR shift names produce identical streams for
        # the same seed; mean_shift and combined likewise match.
        if scenario_key == "mean_shift":

            mean = mean_before if i < change_at else mean_after

        elif scenario_key == "variance_shift":

            mean = mean_before

        elif scenario_key in {"correlation_shift", "covariance_shift"}:

            mean = mean_before
            cov = cov_before if i < change_at else cov_after

        elif scenario_key in {"ar_shift", "autocorrelation_shift"}:

            mean = mean_before

        elif scenario_key == "combined":

            mean = mean_before if i < change_at else mean_after
            cov = cov_before if i < change_at else cov_after

        else:

            mean = mean_before if i < change_at else mean_after
            cov = cov_before if i < change_at else cov_after

        if i == 0:

            # X_0 ~ N(mu,Q): not an AR stationary draw when phi != 0.
            stream[i] = rng.multivariate_normal(mean, cov)

        else:

            # The AR switch applies to every scenario, not only ar_shift.
            phi = ar_prev if i < change_at else ar_post

            # epsilon_t ~ N(0,Q_r): sampled bivariate Gaussian innovation.
            noise = rng.multivariate_normal(
                mean=np.zeros(2),
                cov=cov,
            )

            # Mean-centred recursion: X_t = mu_r + phi_r*(X_(t-1)-mu_r) + epsilon_t.
            stream[i] = (
                phi * (stream[i - 1] - mean)
                + noise
                + mean
            )

    # Configuration metadata, not population moments: Gamma=Q/(1-phi**2).
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

    return stream, metadata
