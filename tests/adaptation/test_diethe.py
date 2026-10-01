import json
import numpy as np
import pandas as pd
import pytest
from src.adaptation.experiments.diethe import DietheConfig, run_synthetic, export_experiment, synthetic_features
from src.adaptation.policies import AdaptationContext, RetrainOnPerformanceDrop, RetrainOnValidatedShift
from src.adaptation.supervised import run_supervised_adaptation


def test_validation_decision_time_and_prediction_before_update():
    class Spy:
        training_size = 2
        version = 0
        def fit(self, x, y): return self
        def predict(self, x): return np.array([self.version])
        def append_and_retrain(self, x, y):
            self.version += 1
            self.training_size += len(x)
    result = run_supervised_adaptation(
        calibration_features=np.array([[-1.], [1.]]), calibration_labels=np.array([0, 1]),
        evaluation_features=np.ones((4, 1)), evaluation_labels=np.ones(4, dtype=int),
        evaluation_times=np.arange(4), classifier=Spy(), policy=RetrainOnValidatedShift(),
        validation_results=pd.DataFrame([dict(alarm_time=0, validation_time=2,
                                             confirmed_shift=True, status='confirmed', p_value=.01)]),
    )
    assert result.trial_results.predicted_label.tolist() == [0, 0, 0, 1]
    assert result.update_events.trigger_time.tolist() == [2]
    assert result.update_events.added_samples.tolist() == [3]
    assert result.trial_results.classifier_version.tolist() == [0, 0, 0, 1]


def test_performance_rule_requires_full_window_and_cooldown():
    rule = RetrainOnPerformanceDrop(.9, .1, 20)
    assert not rule.should_update(AdaptationContext(20, 20, recent_accuracy=None, trials_since_update=21))
    assert not rule.should_update(AdaptationContext(20, 20, recent_accuracy=.5, trials_since_update=1))
    assert not rule.should_update(AdaptationContext(20, 20, recent_accuracy=.8, trials_since_update=21))
    assert rule.should_update(AdaptationContext(20, 20, recent_accuracy=.7, trials_since_update=21))


def test_matched_policy_runs_and_lossless_export(tmp_path):
    result = run_synthetic(DietheConfig(trials=80, scenario='relationship_shift'))
    rows = {r['policy']: r for r in result['summary']}
    assert rows['never']['update_count'] == 0
    assert rows['periodic']['update_count'] == 4
    assert rows['never']['final_training_size'] == 120
    assert rows['periodic']['final_training_size'] == 200
    labels = [r['true_label'] for r in result['runs']['never']['trials']]
    for run in result['runs'].values():
        assert [r['true_label'] for r in run['trials']] == labels
        assert len(run['trials']) == 80
    output = export_experiment(result, tmp_path / 'run')
    saved = json.loads((output / 'experiment.json').read_text())
    assert saved == result
    with pytest.raises(FileExistsError):
        export_experiment(result, output)


def test_scenarios_separate_input_and_relationship_changes():
    base, _ = synthetic_features(DietheConfig(trials=80, scenario='none'))
    relationship, at = synthetic_features(DietheConfig(trials=80, scenario='relationship_shift'))
    assert np.array_equal(base['evaluation_features'], relationship['evaluation_features'])
    assert np.array_equal(base['evaluation_labels'][at:], 1 - relationship['evaluation_labels'][at:])
    features, at = synthetic_features(DietheConfig(trials=80, scenario='feature_shift'))
    x = features['evaluation_features']
    assert np.array_equal(features['evaluation_labels'], (x[:,0] + .5*x[:,1] > 0).astype(int))


@pytest.mark.parametrize('kwargs', [{'trials': 100000}, {'magnitude': float('nan')}, {'accuracy_drop':float('nan')}, {'scenario':'bad'}, {'seed':-1}])
def test_bounded_inputs(kwargs):
    with pytest.raises(ValueError): DietheConfig(**kwargs)
