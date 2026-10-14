# Slide 22: mean, variance and temporal-dependence changes

**PowerPoint location:** Physical slide 22 of `presentations/EWMA_Covariate_Shift_Progress_Review_Haocheng_Lin.pptx`, titled **"Mean validation has a variance-change limitation"**.

## Generate images

From the repository root with the dependencies in `requirements.txt` installed:

```bash
python -m scripts.generate_slide22_structural_visuals
python -m pytest tests/simulation/test_slide22_structural_visuals.py -q
```

Generates:

- `slide22_scenarios.png` and `.svg`: three time-series traces with known breakpoint
- `slide22_rolling_signatures.png` and `.svg`: 3 x 3 matrix of rolling mean, variance and lag-1 correlation
- `slide22_method_targets.png` and `.svg`: conceptual method-target matrix, **NOT measured detection rates**
- `config.json`: simulation parameters and provenance

Defaults: 1024 observations, breakpoint at row 512 (zero-based), rolling window 96, deterministic random seed 42. Plots are presentation-ready PNG (200 dpi) and vector SVG. Optional CLI switches: `--seed`, `--n`, `--change-at`, `--window`, `--output`.

## Statistical meaning

The script uses a **separate controlled generator** rather than `src/simulation/structural_changes.py`. That existing module includes AR coefficient switches in several nominally distinct scenarios and would confound the proposed isolation exercise.

The new scenarios use the stationary-target AR(1) recursion:

```text
X[t] = mu + phi * (X[t-1] - mu) + sqrt(v * (1 - phi**2)) * z[t]
z[t] ~ Normal(0, 1)
```

The pre-change parameters are (mean, stationary variance, lag-1 correlation) = (0, 1, 0.25). Post-change parameters are:

- mean shift: (1.5, 1, 0.25)
- variance shift: (0, 4, 0.25)
- dependence shift: (0, 1, 0.75)

At the transition, observations temporarily depend on the old regime (no resetting). Thus the tuples are **regime stationary targets**, not a claim of instant realised moments at the breakpoint. Sample rolling estimates are noisy.

The matrix identifies the quantities the **conceptual methods** directly test; it does not run Hotelling T², CUSUM or a lag-correlation detector. A mean test can sometimes respond to variance-only changes due to noise, invalid calibration, or other effects; this is not evidence of valid variance-change detection.

**These plots are new controlled illustrations, not Park et al. (2023) Figure 3 reproductions or measured detector-performance results.** The separate Park model A/B/C source in PR #27 covers that paper's specified simulation families.

Slide citation: code-generated figure, `scripts/generate_slide22_structural_visuals.py` (seed 42); theoretical motivation Park et al. (2023), and Chowdhury et al. (2018) Stage-II mean test. Do not attribute the invented simulation parameter values to either paper.

## Repository placement

Keep reusable data generators under `src/simulation/`, executable plot assembly under `scripts/`, tests under `tests/simulation/`, and slide assets plus their `config.json` in this output directory. Source PowerPoint remains unchanged.
