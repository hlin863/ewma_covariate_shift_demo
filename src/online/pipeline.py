"""Full chronological feature-level adaptive BCI controller.

Reuses FBCSP output, frozen PCA/SD-EWMA, paper-aligned two-window Hotelling,
AdaptationPolicy and RetrainableClassifier. Labels are delivered independently
of observation arrivals. No future evaluation labels enter prediction or fitting.
"""
from collections import deque
from dataclasses import dataclass, field
from time import perf_counter

import numpy as np
import pandas as pd

from src.adaptation.classifier import LinearSVMClassifier, RetrainableClassifier
from src.adaptation.policies import (
    AdaptationContext, AdaptationPolicy, RetrainOnValidatedShift,
)
from src.detection.stage2.paper_two_sample_hotelling import PaperTwoSampleHotellingConfig
from src.online.replay import OnlinePC1Monitor, replay_feature_rows
from src.online.validation import OnlineHotellingValidator


@dataclass(frozen=True)
class OnlineAdaptationConfig:
    window_size: int = 10
    alpha: float = 0.05
    covariance_method: str = "shrinkage"
    minimum_alarm_gap: int | None = None
    label_delay: int | None = 0  # None: labels are never released during replay
    update_scope: str = "since_last_update"

    def __post_init__(self) -> None:
        if self.label_delay is not None and self.label_delay < 0:
            raise ValueError("label_delay must be non-negative or None")
        if self.update_scope not in ("since_last_update", "current_trial"):
            raise ValueError("unsupported update_scope")
        PaperTwoSampleHotellingConfig(
            window_size=self.window_size, alpha=self.alpha,
            covariance_method=self.covariance_method,
            minimum_alarm_gap=self.minimum_alarm_gap,
        )


@dataclass(frozen=True)
class OnlineAdaptiveResult:
    trials: pd.DataFrame
    validations: pd.DataFrame
    updates: pd.DataFrame
    final_classifier: RetrainableClassifier
    remaining_pending_warnings: int


class OnlineAdaptiveBCI:
    """Predict -> detect -> validate due warnings -> policy -> label-gated update."""

    def __init__(
        self,
        monitor: OnlinePC1Monitor,
        classifier: RetrainableClassifier,
        policy: AdaptationPolicy | None = None,
        config: OnlineAdaptationConfig | None = None,
    ) -> None:
        self.config = config or OnlineAdaptationConfig()
        self.monitor = monitor
        self.classifier = classifier
        self.policy = policy or RetrainOnValidatedShift()
        self.validator = OnlineHotellingValidator(PaperTwoSampleHotellingConfig(
            window_size=self.config.window_size,
            alpha=self.config.alpha,
            covariance_method=self.config.covariance_method,
            minimum_alarm_gap=self.config.minimum_alarm_gap,
        ))
        self.trials: list[dict[str, object]] = []
        self.validations: list[dict[str, object]] = []
        self.updates: list[dict[str, object]] = []
        self._samples: dict[int, np.ndarray] = {}
        self._labels: dict[int, object] = {}
        self._used_for_training: set[int] = set()
        self._update_requested = False
        self._last_update_index = -1
        self._version = 0
        self._last_time: int | None = None

    @classmethod
    def from_calibration(
        cls,
        calibration_features: np.ndarray,
        calibration_labels: np.ndarray,
        *,
        policy: AdaptationPolicy | None = None,
        config: OnlineAdaptationConfig | None = None,
        pca_components: int | float | None = None,
        lambda_override: float | None = 0.2,
        control_limit_multiplier: float = 1.96,
        variance_update_mode: str = "always",
        classifier: RetrainableClassifier | None = None,
    ) -> "OnlineAdaptiveBCI":
        model = classifier if classifier is not None else LinearSVMClassifier()
        model.fit(calibration_features, calibration_labels)
        monitor = OnlinePC1Monitor.from_training_features(
            calibration_features,
            pca_components=pca_components,
            lambda_override=lambda_override,
            control_limit_multiplier=control_limit_multiplier,
            variance_update_mode=variance_update_mode,
        )
        return cls(monitor, model, policy, config)

    def _attempt_update(self, time: int, reason: str) -> bool:
        if not self._update_requested:
            return False
        available = sorted(
            t for t in self._labels
            if t in self._samples and t not in self._used_for_training
        )
        if self.config.update_scope == "current_trial":
            available = [t for t in available if t == time]
        if not available:
            return False
        features = np.stack([self._samples[t] for t in available])
        labels = np.asarray([self._labels[t] for t in available])
        before = self.classifier.training_size
        started = perf_counter()
        self.classifier.append_and_retrain(features, labels)
        elapsed = perf_counter() - started
        self._used_for_training.update(available)
        for t in available:
            del self._samples[t]
        self._version += 1
        self._last_update_index = len(self.trials) - 1
        self._update_requested = False
        self.updates.append(dict(
            time=int(time), model_version=self._version, reason=reason,
            added_samples=len(available), training_size_before=before,
            training_size_after=self.classifier.training_size,
            retrain_seconds=elapsed, sample_times=available,
        ))
        return True

    def receive_label(self, trial_time: int, label: object, *, available_at: int) -> bool:
        """Release a label only after its availability time, never at observation time by assumption."""
        if self._last_time is None or available_at > self._last_time:
            raise ValueError("label has not arrived yet")
        if trial_time not in self._samples:
            raise ValueError("unknown trial or already used for training")
        if trial_time in self._labels:
            raise ValueError("label already delivered")
        self._labels[trial_time] = label
        return self._attempt_update(self._last_time, "label_arrived")

    def process_one(self, features: np.ndarray, time: int) -> dict[str, object]:
        x = np.asarray(features, dtype=float)
        if x.ndim != 1 or not np.isfinite(x).all():
            raise ValueError("one finite feature vector is required")
        if self._last_time is not None and time <= self._last_time:
            raise ValueError("trial time must increase")
        if time in self._samples or time in self._used_for_training:
            raise ValueError("duplicate trial time")
        # Predict with the existing classifier BEFORE any update on this trial.
        started = perf_counter()
        pred = self.classifier.predict(x.reshape(1, -1))[0]
        prediction_seconds = perf_counter() - started
        monitor_result = self.monitor.update_one(x, time)
        self._last_time = int(time)
        self._samples[int(time)] = x.copy()
        stage1 = monitor_result.stage1_trace
        record: dict[str, object] = dict(
            trial_index=len(self.trials), time=int(time),
            prediction=pred.item() if hasattr(pred, "item") else pred,
            model_version=self._version,
            stage1_warning=bool(stage1["stage_1_alarm"]),
            ewma=float(stage1["ewma"]),
            prediction_seconds=prediction_seconds,
        )
        self.trials.append(record)
        events = self.validator.update_one(
            int(time), monitor_result.projected_features,
            bool(stage1["stage_1_alarm"]),
        )
        for event in events:
            self.validations.append(event)
            if event["status"] == "pending_current_window":
                continue
            context = AdaptationContext(
                trial_index=len(self.trials)-1,
                time=int(time),
                stage1_warning=bool(stage1["stage_1_alarm"]),
                validation_record=event if event.get("validation_time") is not None else None,
                trials_since_update=len(self.trials)-1-self._last_update_index,
            )
            if self.policy.should_update(context):
                self._update_requested = True
        # Warning-based policies must also see current warning even when Stage-II is pending.
        if bool(stage1["stage_1_alarm"]):
            context = AdaptationContext(
                trial_index=len(self.trials)-1,
                time=int(time), stage1_warning=True,
                trials_since_update=len(self.trials)-1-self._last_update_index,
            )
            if self.policy.should_update(context):
                self._update_requested = True
        else:
            context = AdaptationContext(
                trial_index=len(self.trials)-1,
                time=int(time), stage1_warning=False,
                trials_since_update=len(self.trials)-1-self._last_update_index,
            )
            if self.policy.should_update(context):
                self._update_requested = True
        self._attempt_update(time, "policy_trigger")
        record["update_requested"] = self._update_requested
        return record

    def run_replay(
        self,
        evaluation_features: np.ndarray,
        evaluation_times: np.ndarray,
        evaluation_labels: np.ndarray | None = None,
    ) -> OnlineAdaptiveResult:
        labels = None if evaluation_labels is None else np.asarray(evaluation_labels)
        values = np.asarray(evaluation_features, dtype=float)
        if labels is not None and (labels.ndim != 1 or labels.size != len(values)):
            raise ValueError("evaluation labels must match evaluation observations")
        if labels is not None and self.config.label_delay is None:
            # No labels are made available to the learner; labels remain scoring-only.
            pass
        due: deque[tuple[int, int, object]] = deque()
        for index, (time, x) in enumerate(replay_feature_rows(values, evaluation_times)):
            self.process_one(x, time)
            if labels is not None and self.config.label_delay is not None:
                due.append((index+self.config.label_delay, time, labels[index]))
            while due and due[0][0] <= index:
                _, trial_time, label = due.popleft()
                self.receive_label(trial_time, label, available_at=time)
        for event in self.validator.finish():
            # The latest status for the same warning is pending at end of stream.
            self.validations.append(event)
        trials = pd.DataFrame(self.trials)
        if labels is not None:
            # Scoring is post-run only; do not expose truth during prediction.
            truth_by_time = {int(t): y for t, y in zip(evaluation_times, labels)}
            trials["true_label"] = trials["time"].map(truth_by_time)
            trials["correct"] = trials["prediction"] == trials["true_label"]
        return OnlineAdaptiveResult(
            trials=trials,
            validations=pd.DataFrame(self.validations),
            updates=pd.DataFrame(self.updates),
            final_classifier=self.classifier,
            remaining_pending_warnings=len(self.validator.pending),
        )
