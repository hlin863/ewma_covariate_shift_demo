import numpy as np
import pandas as pd

from src.adaptation.policies import RetrainOnValidatedShift
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
