"""Controlled policy comparisons inspired by Diethe (2019), not a reproduction.

Prepared EEG features can use the same API. Labels become available immediately
AFTER each prediction. Detector records are acted on at validation time.
"""
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import subprocess
from time import perf_counter

import numpy as np
import pandas as pd

from src.adaptation.classifier import LinearSVMClassifier
from src.adaptation.policies import (
    NeverUpdate, PeriodicRetrain, RetrainOnWarning, RetrainOnValidatedShift,
    RetrainOnPerformanceDrop,
)
from src.adaptation.supervised import SupervisedAdaptationConfig, run_supervised_adaptation
from src.detection import CSEConfig, run_cse


@dataclass(frozen=True)
class DietheConfig:
    seed: int = 42
    scenario: str = 'feature_shift'
    trials: int = 240
    magnitude: float = 2.0
    interval: int = 20
    performance_window: int = 30
    accuracy_drop: float = 0.10
    update_scope: str = 'since_last_update'

    def __post_init__(self):
        if self.scenario not in {'none', 'feature_shift', 'relationship_shift'}:
            raise ValueError('Unknown scenario.')
        if not 0 <= self.seed <= 2**32 - 1:
            raise ValueError('Seed must be between 0 and 4294967295.')
        if not 80 <= self.trials <= 500:
            raise ValueError('Trials must be between 80 and 500.')
        if not np.isfinite(self.magnitude) or not 0 <= self.magnitude <= 5:
            raise ValueError('Magnitude must be between 0 and 5.')
        if not 5 <= self.interval <= 100 or not 5 <= self.performance_window <= 100:
            raise ValueError('Interval and performance window must be between 5 and 100.')
        if not np.isfinite(self.accuracy_drop) or not 0 < self.accuracy_drop <= 1:
            raise ValueError('Accuracy drop must be in (0, 1].')
        SupervisedAdaptationConfig(self.update_scope, self.performance_window)


def synthetic_features(config):
    rng = np.random.default_rng(config.seed)
    train = rng.normal(size=(120, 3))
    validation = rng.normal(size=(80, 3))
    evaluation = rng.normal(size=(config.trials, 3))
    shift_at = config.trials // 2
    if config.scenario == 'feature_shift':
        evaluation[shift_at:, 0] += config.magnitude
    def labels(x):
        return (x[:, 0] + 0.5 * x[:, 1] > 0).astype(int)
    y_eval = labels(evaluation)
    if config.scenario == 'relationship_shift':
        y_eval[shift_at:] = 1 - y_eval[shift_at:]
    return dict(training_features=train, training_labels=labels(train),
                validation_features=validation, validation_labels=labels(validation),
                evaluation_features=evaluation, evaluation_labels=y_eval,
                evaluation_times=np.arange(config.trials)), (
                    None if config.scenario == 'none' else shift_at)


def _records(frame):
    return json.loads(frame.to_json(orient='records'))


def run_policy_comparison(*, training_features, training_labels,
                          validation_features, validation_labels,
                          evaluation_features, evaluation_labels, evaluation_times,
                          config=None, source='prepared_features', change_index=None,
                          detector_config=None, provenance=None):
    config = config or DietheConfig()
    x_train, x_val, x_eval = [np.asarray(x, dtype=float) for x in
                            (training_features, validation_features, evaluation_features)]
    y_train, y_val, y_eval = [np.asarray(y) for y in
                            (training_labels, validation_labels, evaluation_labels)]
    times = np.asarray(evaluation_times)
    for x, y in ((x_train, y_train), (x_val, y_val), (x_eval, y_eval)):
        if x.ndim != 2 or len(x) == 0 or y.ndim != 1 or len(x) != len(y):
            raise ValueError('Each feature matrix needs matching one-dimensional labels.')
        if not np.isfinite(x).all() or not np.isin(y, [0, 1]).all():
            raise ValueError('Features must be finite and labels encoded as 0/1.')
    if len(set(x.shape[1] for x in (x_train, x_val, x_eval))) != 1:
        raise ValueError('Feature dimensions must match.')
    if len(np.unique(y_train)) != 2:
        raise ValueError('Training requires both classes.')
    if (times.ndim != 1 or len(times) != len(x_eval) or
            not np.isfinite(times).all() or not np.equal(times, np.floor(times)).all()
            or np.any(np.diff(times) <= 0)):
        raise ValueError('Evaluation times must be strictly increasing integer trial indices.')
    if change_index is not None and not 0 < change_index < len(x_eval):
        raise ValueError('Change index must be inside the evaluation stream.')

    baseline = LinearSVMClassifier().fit(x_train, y_train)
    reference_accuracy = float(np.mean(baseline.predict(x_val) == y_val))
    detector_config = detector_config or CSEConfig(
        pca_components=min(3, x_train.shape[1]), lambda_override=0.2,
        control_limit_multiplier=2.5, variance_update_mode='frozen',
        validation_mode='paper_two_sample', validation_before_size=20,
        validation_after_size=20, covariance_method='empirical',
    )
    started = perf_counter()
    detection = run_cse(x_train, x_eval, times, detector_config)
    detector_seconds = perf_counter() - started
    policies = {
        'never': NeverUpdate(), 'periodic': PeriodicRetrain(config.interval),
        'warning': RetrainOnWarning(), 'validated': RetrainOnValidatedShift(),
        'performance_drop': RetrainOnPerformanceDrop(
            reference_accuracy, config.accuracy_drop, config.interval),
    }
    runs, summaries = {}, []
    for name, policy in policies.items():
        result = run_supervised_adaptation(
            calibration_features=x_train, calibration_labels=y_train,
            evaluation_features=x_eval, evaluation_labels=y_eval,
            evaluation_times=times, validation_results=detection.validation_results,
            warning_results=detection.warning_results, policy=policy,
            config=SupervisedAdaptationConfig(config.update_scope, config.performance_window),
        )
        trials = result.trial_results
        retrain_seconds = (float(result.update_events.retrain_seconds.sum())
                           if not result.update_events.empty else 0.0)
        summaries.append(dict(
            policy=name, accuracy=result.accuracy, update_count=result.update_count,
            initial_fit_seconds=result.initial_fit_seconds,
            retrain_seconds=retrain_seconds,
            prediction_seconds=float(trials.prediction_seconds.sum()),
            final_training_size=result.final_classifier.training_size,
            post_change_accuracy=(float(trials.correct.iloc[change_index:].mean())
                                  if change_index is not None else None),
        ))
        runs[name] = dict(trials=_records(trials), updates=_records(result.update_events))
    baseline_accuracy = summaries[0]['accuracy']
    for row in summaries:
        row['accuracy_gain_pp'] = 100 * (row['accuracy'] - baseline_accuracy)
    try:
        commit = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).resolve().parents[3],
            stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = 'unavailable (exported source checkout)'
    return dict(
        metadata=dict(source=source, source_commit=commit, provenance=provenance or {},
                      config=asdict(config), detector_config=asdict(detector_config),
                      evaluation_trials=len(x_eval), training_trials=len(x_train),
                      validation_trials=len(x_val), change_index=change_index,
                      label_protocol='Immediate label after prediction; no delayed-label simulation',
                      reference_accuracy=reference_accuracy,
                      pc1_variance_ratio=float(detection.pca_result.explained_variance_ratio[0]),
                      detector_seconds=detector_seconds,
                      boundary='Fixed feature representation and detector; expanding SVM training; no rollback'),
        summary=summaries, runs=runs,
        warnings=_records(detection.warning_results),
        validations=_records(detection.validation_results),
        validation_status_counts={str(k): int(v) for k, v in
                                  detection.validation_results.status.value_counts().items()},
    )


def run_synthetic(config=None):
    config = config or DietheConfig()
    features, change_index = synthetic_features(config)
    return run_policy_comparison(**features, config=config,
                                 source='synthetic feature vectors (not EEG)',
                                 change_index=change_index)


def export_experiment(result, output):
    """Never overwrite an existing run; save event evidence and provenance."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'experiment.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    pd.DataFrame(result['summary']).to_csv(output / 'summary.csv', index=False)
    pd.DataFrame(result['warnings']).to_csv(output / 'warnings.csv', index=False)
    pd.DataFrame(result['validations']).to_csv(output / 'validations.csv', index=False)
    for policy, records in result['runs'].items():
        pd.DataFrame(records['trials']).to_csv(output / f'{policy}_trials.csv', index=False)
        pd.DataFrame(records['updates']).to_csv(output / f'{policy}_updates.csv', index=False)
    return output
