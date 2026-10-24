# ABCD Experiment Lab

Paper: Heyden et al. (2024), https://doi.org/10.1007/s10618-023-00999-5
Official implementation: https://github.com/heymarco/AdaptiveBernsteinChangeDetector
Stream generators: https://github.com/heymarco/StreamGenerators

## Scope and equations

This is an equation-level **teaching baseline**, not a reproduction of the published results and not the authors' ABCD Algorithm 2.

- Uses two Gaussian concepts (200 observations by default), not ten concepts × 2,000 observations and 30 repetitions.
- PCA is trained on the first concept, frozen, and used to reconstruct observations. Mean squared reconstruction loss corresponds to Eq. (1).
- Uses Eq. (5) via src/detection/baselines/abcd.py, approximate kappa selection from Eq. (12), and the candidate minimum of Eq. (8).
- The paper's synthetic Gaussian generator is used as conceptual inspiration only: dimensions, change magnitude, sequence length and preprocessing are controlled here.
- Global MinMax scaling is a deliberate parallel to the authors' preprocess function but leaks future data and is not a defensible online protocol.
- Sample variance is substituted for population variance, and repeated split testing has no sequential false-alarm correction. The returned score is a bound-based teaching score, not a calibrated stream-wide p-value.
- M=0.1 is the empirical setting discussed in the paper, not a generally valid almost-sure deviation bound. M=1 is conservative when both inputs and reconstructions lie in [0,1].

## Web integration

The existing src/web/__init__.py registers the new Blueprint; the existing app.py entry point remains unchanged.
URL: http://127.0.0.1:5000/results/abcd
The page is linked in the existing Results menu.

Commands from repository root:

    python -m pytest tests/experiments/test_abcd_lab.py tests/web/test_abcd_page.py -q
    python app.py

Use the form to change scenario, seed, feature dimensions, shift magnitude, bottleneck, M, delta and the number of candidate split locations.
The interactive figures and table display per-observation losses, candidate split bounds, term-level evidence and an exportable JSON run.

## Next stage: reference reproduction

The original author's E_synthetic_data.py uses StreamGenerators.Gaussian, Hypersphere and LED.
For matched evaluation, recreate the authors' exact generators and parameter grid, original ABCD AdaptiveWindow and online representation updates, aggregation, severity/subspace analysis, and precision/recall/F1/MTD evaluation.

HAR and Gas Sensor Drift experiments also require the upstream raw datasets. Avoid claiming that this limited synthetic teaching experiment reproduces any reported results.
