import json
import numpy as np
import pytest
from src.adaptation.experiments.diethe import DietheConfig, run_synthetic, export_experiment, synthetic_features


def test_matched_policy_runs_and_lossless_export(tmp_path):
    result = run_synthetic(DietheConfig(trials=80, scenario='relationship_shift'))
    rows = {r['policy']: r for r in result['summary']}
    assert rows['never']['update_count'] == 0
    assert rows['periodic']['update_count'] == 4
    assert rows['never']['policy_family'] == 'non-adaptive'
    assert rows['periodic']['policy_family'] == 'passive'
    assert rows['validated']['policy_family'] == 'active'
    assert rows['periodic']['trigger'] == '20 new trials since previous update'
    assert rows['never']['final_training_size'] == 120
    assert rows['periodic']['final_training_size'] == 200
    assert [event['trials_since_update'] for event in result['runs']['periodic']['updates']] == [20, 20, 20, 20]
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
