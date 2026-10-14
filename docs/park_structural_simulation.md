# Park et al. (2023): Section 3.1 simulation sources

See [notebook 06](../notebooks/06_park_structural_model_simulations.ipynb), production [generator](../src/simulation/park_structural_models.py), and [tests](../tests/simulation/test_park_structural_models.py).

The Figure 3 reference shows six panels (three model families, each with and without a change). The model parameters are derived from Section 3.1; the fixed random seeds and exact displayed channel are reconstruction choices, **not verified author settings**.

- **A:** AR(1), alpha drawn from U(-0.5, 0.5), sigma from U(0.5, 2), innovation N(0,4). Selected channels (floor(rho*p)) regenerate alpha and sigma and shift innovation means using mu drawn from U(2,5).
- **B(i):** AR(1), initial alpha drawn from U(0.5,1), innovation N(0,1); selected channels change alpha to draws from U(-0.5,0.3).
- **C:** MA(1), initial alpha drawn from U(0.5,1), innovation N(0,1); selected channels change alpha to draws from U(-0.5,0.3).
- **Figure setting:** 1024 observations, 50 channels, change at index 512. Supported sparsities rho = 0.7, 0.5, 0.3, 0.2, 0.1. The paper also evaluates 512 observations and alternative breakpoints.
- **Boundary convention:** first changed observation at zero-based Python row 512; initial x[0] uses the same model with zero pre-sample history.
- **Excluded:** Model B(ii) (correlated multivariate innovations and rearranged cross-dependence), which should be implemented as an explicitly separate extension.
- **Detector compatibility:** signed series must be retained for faithful model simulation. The current mean-CUSUM method requires positive channel means, and B/C are second-order changes. Do not apply arbitrary offsets or squares and call the resulting detector behaviour a reproduction of Park's mean-change decision procedure.
- **Reproduction limit:** generated data statistically follow the specified model families but cannot match the exact random traces in Figure 3 without the authors' original random generator state.

Run tests with: python -m pytest tests/simulation/test_park_structural_models.py -q
