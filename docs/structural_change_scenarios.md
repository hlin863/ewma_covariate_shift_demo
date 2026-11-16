# Isolated multivariate structural-change simulations

The two-feature synthetic generator is `src/simulation/structural_changes.py`.
These scenarios are exploratory and **not** a reproduction of Park et al. (2023)
second-order wavelet/DPCA change detection.

## Model and target statistics

The centered AR(1) recursion is `X_t = mu_r + phi_r (X_{t-1}-mu_r) + epsilon_t`,
where `epsilon_t ~ N(0, Q_r)` and the target stationary covariance is

`Sigma_r = v_r [[1,rho_r],[rho_r,1]]`.

For a scalar `phi_r`, use `Q_r=(1-phi_r**2)*Sigma_r`. This gives
`Cov(X_t,X_{t-h})=phi_r**h * Sigma_r` within a stationary regime.
A change to `phi` can therefore preserve the zero-lag covariance while
changing lagged dependence. Immediately following a regime transition the
process inherits its previous state; the post-change stationary covariance
is a *target*, not necessarily the instantaneous covariance of the first
post-change sample.

## Scenarios

| Scenario | Mean | Stationary marginal variance | Cross-correlation | AR coefficient |
| --- | --- | --- | --- | --- |
| stationary | fixed | fixed | fixed | fixed |
| mean_shift | changed | fixed | fixed | fixed |
| variance_shift | fixed | changed | fixed | fixed |
| correlation_shift | fixed | fixed | changed | fixed |
| ar_shift | fixed | fixed | fixed | changed |
| combined | changed | changed | changed | changed |

For compatibility, `covariance_shift` aliases `correlation_shift`,
`autocorrelation_shift` aliases `ar_shift`, and historical `default`
aliases `combined`. Invalid scenario names raise a `ValueError`.

The `variance_ratio` parameter now describes the *stationary marginal
variance* ratio, whereas older runs changed innovation covariance with
a simultaneously changed AR parameter. Do **not** compare new numerical
traces with archived output under the assumption of identical semantics.

## Provenance and tests

The metadata now includes `resolved_scenario`, `before`, and `after`
with explicit means, AR coefficients, target covariance and innovation
covariance matrices. Keep prior archived visualizations intact; new runs
should be generated under new run IDs.

Run `python -m pytest tests/simulation/test_structural_changes.py
tests/simulation/test_slide22_structural_visuals.py -q`.

The separate univariate slide generator
`scripts/generate_slide22_structural_visuals.py` already used
stationary-target isolation, and its tests supplied the precedent.

The corrected generator enables valid controlled probes but by itself
does **not** implement a second-order change-point detector, validate
an online false-alarm rate, or establish any BCI adaptation gain.
