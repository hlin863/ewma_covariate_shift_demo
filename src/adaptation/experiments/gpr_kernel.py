"""Controlled diagnostic experiments; no classifier retraining is performed."""
from dataclasses import asdict
import numpy as np
from src.detection.stage1.sd_ewma import SD_EWMA_Config, fit_sd_ewma, run_sd_ewma
from src.detection.stage2.gpr_kernel_hotelling import (
    GPRKernelConfig, fit_gpr_kernel, validate_gpr_kernel_alarms,
)

SCENARIOS = ('none', 'mean_shift', 'variance_shift', 'support_shift', 'relationship_shift')


def run_prepared(training_features, training_targets, calibration_features,
                 evaluation_features, evaluation_times, config=None):
    config = config or GPRKernelConfig()
    model = fit_gpr_kernel(training_features, training_targets, calibration_features, config)
    # Fixed first-feature EWMA warning route. Do not inject a known change-point warning.
    train = model.scaler.transform(training_features)[:, 0]
    values = model.scaler.transform(evaluation_features)[:, 0]
    ewma = fit_sd_ewma(train, lambda_override=0.2)
    warnings = run_sd_ewma(values=values, times=evaluation_times,
        initial_z=ewma.final_z, initial_error_variance=ewma.error_variance,
        config=SD_EWMA_Config(lambda_value=0.2, control_limit_multiplier=2.5,
                             variance_update_mode='frozen'))
    validations = validate_gpr_kernel_alarms(evaluation_features, evaluation_times, warnings, model)
    return {
        'config': asdict(config), 'gamma': model.gamma,
        'uncertainty_limit': model.uncertainty_limit,
        'gpr_kernel': str(model.regressor.kernel_),
        'warning_route': 'first standardized feature; EWMA lambda=0.2, L=2.5, frozen variance',
        'warnings': warnings.replace({np.nan: None}).to_dict('records'),
        'validations': validations.replace({np.nan: None}).to_dict('records'),
        'summary': {'warnings': int(warnings.stage_1_alarm.sum()),
            'evaluated_diagnostics': int((validations.diagnostic_status == 'evaluated').sum()),
            'hotelling_confirmations': int(validations.confirmed_shift.sum()),
            'kernel_confirmations': int(validations.kernel_shift.sum()),
            'high_uncertainty_windows': int(validations.high_uncertainty.sum()),
            'adaptation_recommendations': int(validations.adaptation_recommended.sum())},
        'boundary': 'Synthetic/diagnostic verification, not EEG reproduction or adaptation benefit. '
                    'Permutation p-values assume IID windows; multiple warnings have no stream-wide alpha guarantee.',
    }


def run_synthetic(scenario='mean_shift', seed=42, permutations=199):
    if scenario not in SCENARIOS:
        raise ValueError('Unknown scenario.')
    config = GPRKernelConfig(seed=seed, permutations=permutations)
    rng = np.random.default_rng(seed)
    training, calibration, evaluation = [rng.normal(size=(n, 3)) for n in (120, 200, 200)]
    target = np.sin(training[:, 0]) + 0.5 * training[:, 1] + rng.normal(0, 0.1, len(training))
    change = 100
    if scenario == 'mean_shift':
        evaluation[change:, 0] += 2
    elif scenario == 'variance_shift':
        evaluation[change:, 0] *= 3
    elif scenario == 'support_shift':
        evaluation[change:, :2] += 4
    # Relationship scenario leaves features unchanged and reverses a hypothetical
    # target function; no evaluation targets are available to these detectors.
    result = run_prepared(training, target, calibration, evaluation, np.arange(200), config)
    result['source'] = {'dataset': 'IID synthetic continuous regression', 'scenario': scenario,
        'seed': seed, 'intervention_index': None if scenario == 'none' else change,
        'training_rows': 120, 'calibration_rows': 200, 'evaluation_rows': 200,
        'relationship_change_observable': False}
    return result
