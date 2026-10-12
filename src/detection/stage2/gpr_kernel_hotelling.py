"""Opt-in diagnostics beside unchanged equal-window Hotelling validation.

GPR predicts a supplied continuous target (or a binary-label surrogate), not
class probabilities. Uncertainty is latent-function posterior standard deviation
with observation noise supplied through alpha, not a WhiteKernel.
"""
from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import pdist
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import StandardScaler

from src.detection.stage2.paper_two_sample_hotelling import (
    PaperTwoSampleHotellingConfig, validate_paper_two_sample_alarms,
)


def _features(values, name):
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or min(array.shape) == 0 or not np.isfinite(array).all():
        raise ValueError(f'{name} must be a nonempty finite (n, D) array.')
    return array


@dataclass(frozen=True)
class GPRKernelConfig:
    window_size: int = 20
    alpha: float = 0.05
    permutations: int = 199
    seed: int = 42
    noise_variance: float = 0.1
    length_scale: float = 1.0
    uncertainty_quantile: float = 0.95

    def __post_init__(self):
        if not isinstance(self.window_size, int) or not 2 <= self.window_size <= 100:
            raise ValueError('window_size must be an integer in [2, 100].')
        if not 0 < self.alpha < 1 or not 0 < self.uncertainty_quantile < 1:
            raise ValueError('alpha and uncertainty_quantile must be in (0, 1).')
        if not isinstance(self.permutations, int) or not 19 <= self.permutations <= 999:
            raise ValueError('permutations must be an integer in [19, 999].')
        if not isinstance(self.seed, int) or not 0 <= self.seed < 2**32:
            raise ValueError('seed must be an integer in [0, 2**32).')
        if not np.isfinite(self.noise_variance) or self.noise_variance <= 0:
            raise ValueError('noise_variance must be finite and positive.')
        if not np.isfinite(self.length_scale) or self.length_scale <= 0:
            raise ValueError('length_scale must be finite and positive.')


@dataclass
class GPRKernelModel:
    scaler: StandardScaler
    regressor: GaussianProcessRegressor
    gamma: float
    uncertainty_limit: float
    config: GPRKernelConfig


def fit_gpr_kernel(training_features, training_targets, calibration_features,
                   config=None):
    """Fit only on training; set a WINDOW-level std limit on disjoint holdout.

    Caller owns split provenance. Calibration uses nonoverlapping windows;
    neither evaluation features nor labels set hyperparameters or limits.
    """
    config = config or GPRKernelConfig()
    training = _features(training_features, 'training_features')
    calibration = _features(calibration_features, 'calibration_features')
    targets = np.asarray(training_targets, dtype=float)
    if training.shape[1] != calibration.shape[1]:
        raise ValueError('Training and calibration feature counts must match.')
    if targets.shape != (len(training),) or not np.isfinite(targets).all():
        raise ValueError('training_targets must be finite and match training rows.')
    if not 2 <= len(training) <= 500:
        raise ValueError('Use 2–500 training rows for this bounded GPR prototype.')
    h = config.window_size
    if len(calibration) < 5 * h:
        raise ValueError('Calibration requires at least five complete windows.')
    scaler = StandardScaler().fit(training)
    x = scaler.transform(training)
    distances = pdist(x, metric='sqeuclidean')
    positive = distances[distances > 0]
    if not len(positive):
        raise ValueError('Training needs nonzero feature variation.')
    gamma = float(1 / np.median(positive))
    gp = GaussianProcessRegressor(kernel=RBF(config.length_scale), optimizer=None,
                                 alpha=config.noise_variance, normalize_y=True)
    gp.fit(x, targets)
    _, std = gp.predict(scaler.transform(calibration), return_std=True)
    means = std[:len(std) // h * h].reshape(-1, h).mean(axis=1)
    limit = float(np.quantile(means, config.uncertainty_quantile, method='higher'))
    return GPRKernelModel(scaler, gp, gamma, limit, config)


def kernel_mmd(reference, current, *, gamma, permutations=199, seed=42):
    """Unbiased RBF MMD² with a plus-one, IID label-permutation p-value.

    Negative unbiased estimates are retained. Gamma must be fixed independently
    of tested windows. Temporal dependence requires a different null protocol.
    """
    reference = _features(reference, 'reference')
    current = _features(current, 'current')
    if reference.shape[1] != current.shape[1] or min(len(reference), len(current)) < 2:
        raise ValueError('MMD needs matching features and at least two rows per sample.')
    if not np.isfinite(gamma) or gamma <= 0 or permutations < 1:
        raise ValueError('gamma and permutations must be positive.')
    n, m = len(reference), len(current)
    gram = rbf_kernel(np.vstack([reference, current]), gamma=gamma)

    def statistic(order):
        a, b = order[:n], order[n:]
        aa, bb, ab = gram[np.ix_(a, a)], gram[np.ix_(b, b)], gram[np.ix_(a, b)]
        return float((aa.sum() - np.trace(aa)) / (n * (n - 1))
                     + (bb.sum() - np.trace(bb)) / (m * (m - 1)) - 2 * ab.mean())

    order = np.arange(n + m)
    observed = statistic(order)
    rng = np.random.default_rng(seed)
    exceed = sum(statistic(rng.permutation(order)) >= observed for _ in range(permutations))
    return observed, float((exceed + 1) / (permutations + 1))


def validate_gpr_kernel_alarms(features, times, stage_1_results, model):
    """Add diagnostics to baseline records; never overwrite confirmed_shift.

    The exploratory review gate is (T² OR MMD significant) AND high std.
    It is not a joint level-alpha test, an automatic retrain, or evidence of
    changed P(y|x). Pending/skipped warnings remain ineligible.
    """
    features = _features(features, 'features')
    if features.shape[1] != model.scaler.n_features_in_:
        raise ValueError('Evaluation feature count differs from fitted training.')
    config = model.config
    records = validate_paper_two_sample_alarms(features, times, stage_1_results,
        PaperTwoSampleHotellingConfig(window_size=config.window_size, alpha=config.alpha))
    for name in ['mmd_squared', 'mmd_p_value', 'gpr_mean_std']:
        records[name] = np.nan
    records['uncertainty_limit'] = model.uncertainty_limit
    records['kernel_shift'] = False
    records['high_uncertainty'] = False
    records['adaptation_recommended'] = False
    records['diagnostic_status'] = 'unavailable'
    times = np.asarray(times)
    transformed = model.scaler.transform(features)
    for row, record in records.iterrows():
        # Kernel diagnostics can still run if T² has insufficient degrees of freedom.
        i = int(np.flatnonzero(times == record.alarm_time)[0])
        h = config.window_size
        if record.status in {'pending_current_window', 'insufficient_reference_window', 'skipped_nearby_alarm'}:
            continue
        before, after = transformed[i-h+1:i+1], transformed[i+1:i+1+h]
        if len(before) != h or len(after) != h:
            continue
        mmd, p_value = kernel_mmd(before, after, gamma=model.gamma,
                                 permutations=config.permutations, seed=config.seed + i)
        _, std = model.regressor.predict(after, return_std=True)
        mean_std = float(std.mean())
        kernel_shift = p_value < config.alpha
        uncertain = mean_std > model.uncertainty_limit
        records.loc[row, ['mmd_squared', 'mmd_p_value', 'gpr_mean_std']] = [mmd, p_value, mean_std]
        records.loc[row, ['kernel_shift', 'high_uncertainty', 'adaptation_recommended']] = [
            kernel_shift, uncertain, (bool(record.confirmed_shift) or kernel_shift) and uncertain]
        records.loc[row, 'validation_time'] = times[i+h]
        records.loc[row, 'diagnostic_status'] = 'evaluated'
    return records
