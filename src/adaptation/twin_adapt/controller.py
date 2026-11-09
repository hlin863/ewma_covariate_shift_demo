"""Batchwise operational TWIN-ADAPT with delayed-label-safe decisions.

Paper §5.1 Algorithm 1 ambiguities resolved explicitly:
- adapt-window capacity is TRIAL count; duplicates disallowed.
- one state transition per RELEASED batch; both warning/adaptation thresholds
  may be checked on that same batch (paper's independent if statements).
- after initial update, buffer resets and state 2 collects NEW concept data.
- paper's 'f = CurrAcc' cannot index a window; we anchor state-2 reference
  to the first fully labelled post-update batch instead.
- in state 2 update on a further relative drop or buffer reaching capacity.
- full predictions made with the pre-update model, then labels released later.
This is a DOCUMENTED OPERATIONAL VARIANT, not a literal paper reproduction.
"""
from collections import deque
from dataclasses import dataclass
from time import perf_counter
from typing import Iterable
import numpy as np
import pandas as pd
from src.online.batch_stream import BCIBatch
from src.adaptation.twin_adapt.accuracy_windows import calculate_batch_accuracies
from src.adaptation.twin_adapt.adaptive_buffer import AdaptiveBuffer
from src.adaptation.twin_adapt.config import TwinAdaptConfig
from src.adaptation.twin_adapt.evaluation import TwinAdaptResult


@dataclass(frozen=True)
class _PendingFeedback:
    due_batch: int
    batch: BCIBatch
    predictions: np.ndarray
    model_version: int


def adapt_model(dataflow: Iterable[BCIBatch], model: object, *,
                config: TwinAdaptConfig, evaluation_labels: np.ndarray | None) -> TwinAdaptResult:
    """Consume batches and evaluate accuracy only as labels are released.

    evaluation_labels is an OFFLINE REPLAY oracle, indexed by original trial time.
    It is never accessed until the feedback due_batch. `model` must already
    be fitted and implement RetrainableClassifier (predict, training_size,
    append_and_retrain). `dataflow` must yield consecutive unique trial times
    0..N-1, as in the existing `prepare` feature matrices.
    """
    labels = None if evaluation_labels is None else np.asarray(evaluation_labels)
    if labels is not None and labels.ndim != 1:
        raise ValueError("evaluation_labels must be a one-dimensional array")
    state = 0
    version = 0
    buffer = AdaptiveBuffer(config.win_size_max)
    pending: deque[_PendingFeedback] = deque()
    trial_records: list[dict] = []
    window_records: list[dict] = []
    transitions: list[dict] = []
    buffer_records: list[dict] = []
    updates: list[dict] = []
    state2_reference: float | None = None
    last_time = -1
    batch_count = 0

    def update(batch: BCIBatch, reason: str) -> None:
        nonlocal version
        x_new, y_new, selected = buffer.arrays()
        before = int(model.training_size)
        started = perf_counter()
        model.append_and_retrain(x_new, y_new)
        elapsed = perf_counter() - started
        version += 1
        updates.append(dict(batch=batch.batch_index, time=int(batch.times[-1]),
                            reason=reason, version=version, added_samples=len(y_new),
                            selected_trials=selected, training_size_before=before,
                            training_size_after=int(model.training_size),
                            retrain_seconds=elapsed))
        buffer.clear()

    def release(item: _PendingFeedback, release_batch: int) -> None:
        nonlocal state, state2_reference
        batch = item.batch
        y = labels[batch.times]  # type: ignore[index]  # read ONLY at release time
        prev_state = state
        prev_acc = curr_acc = None
        status = "ineligible_short_batch"
        action = "none"
        dropped = 0
        if len(batch.times) > config.win_size:
            prev_acc, curr_acc = calculate_batch_accuracies(
                item.predictions, y, win_size=config.win_size)
            warn_boundary, adapt_boundary = config.gamma*prev_acc, config.delta*prev_acc
            status = "evaluated"
            if state == 0:
                if curr_acc < warn_boundary:
                    state = 1
                    dropped = buffer.add(batch, y)
                    action = "warning_started"
                    # Like paper lines 9 and 13-19, severe drop can update
                    # on the same batch, without appending it twice.
                    if curr_acc < adapt_boundary:
                        update(batch, "initial_severe_accuracy_drop")
                        state = 2
                        state2_reference = None
                        action = "initial_update"
                    elif len(buffer) >= config.win_size_max:
                        buffer.clear()
                        state = 0
                        action = "warning_timeout_no_update"
            elif state == 1:
                if curr_acc < adapt_boundary:
                    dropped = buffer.add(batch, y)
                    update(batch, "confirmed_accuracy_drop")
                    state = 2
                    state2_reference = None
                    action = "initial_update"
                elif curr_acc > warn_boundary:
                    buffer.clear()
                    state = 0
                    action = "recovered_without_update"
                else:
                    dropped = buffer.add(batch, y)
                    action = "collect_warning"
                    if len(buffer) >= config.win_size_max:
                        buffer.clear()
                        state = 0
                        action = "warning_timeout_no_update"
            else:  # state 2: monitor post-update concept, collect fresh batches
                if state2_reference is None:
                    state2_reference = curr_acc
                dropped = buffer.add(batch, y)
                action = "collect_new_concept"
                if (len(buffer) >= config.win_size_max
                        or (state2_reference > 0 and curr_acc < config.delta*state2_reference)):
                    reason = ("new_concept_capacity" if len(buffer) >= config.win_size_max
                              else "new_concept_accuracy_drop")
                    update(batch, reason)
                    state, state2_reference = 0, None
                    action = "new_concept_update"
            window_records.append(dict(batch=batch.batch_index, released_at_batch=release_batch,
                                       prev_accuracy=prev_acc, curr_accuracy=curr_acc,
                                       gamma=config.gamma, delta=config.delta,
                                       warning_threshold=warn_boundary,
                                       adaptation_threshold=adapt_boundary,
                                       original_model_version=item.model_version,
                                       reference_new_concept_accuracy=state2_reference,
                                       action=action))
        if state != prev_state:
            transitions.append(dict(batch=batch.batch_index, released_at_batch=release_batch,
                                    old_state=prev_state, new_state=state, action=action))
        buffer_records.append(dict(batch=batch.batch_index, released_at_batch=release_batch,
                                   buffer_size=len(buffer), dropped_at_capacity=dropped,
                                   state=state, status=status, action=action))

    for batch in dataflow:
        if batch.batch_index != batch_count:
            raise ValueError("Batch indices must start at zero and increase by one")
        indices = np.asarray(batch.times)
        if (indices.ndim != 1 or len(indices) < 1 or
                indices.dtype.kind not in "iu" or
                np.any(np.diff(indices) != 1) or int(indices[0]) != last_time + 1):
            raise ValueError("Expected consecutive chronological trial indices starting at 0")
        if len(indices) > config.batch_size:
            raise ValueError("Stream batch exceeds configured batch_size")
        if labels is not None and int(indices[-1]) >= len(labels):
            raise ValueError("Not enough evaluation labels for replay stream")
        last_time = int(indices[-1])
        predictions = np.asarray(model.predict(batch.features))
        if predictions.shape != (len(indices),):
            raise ValueError("Classifier predictions must match batch length")
        for t, pred in zip(indices, predictions):
            trial_records.append(dict(trial_index=int(t), batch=batch.batch_index,
                                      prediction=pred.item() if hasattr(pred, "item") else pred,
                                      model_version=version))
        if labels is not None and config.label_delay is not None:
            pending.append(_PendingFeedback(batch.batch_index + config.label_delay,
                                            batch, predictions.copy(), version))
        # RELEASE happens only after this whole batch was predicted.
        while pending and pending[0].due_batch <= batch.batch_index:
            release(pending.popleft(), batch.batch_index)
        batch_count += 1
    if batch_count == 0:
        raise ValueError("DataFlow must contain at least one batch")
    trials = pd.DataFrame(trial_records)
    if labels is not None:
        # Post-run scoring: never fed back into controller until release.
        if len(labels) != len(trials):
            raise ValueError("Evaluation label count does not equal streamed trial count")
        trials["true_label"] = labels
        trials["correct"] = trials.prediction.to_numpy() == labels
    else:
        trials["correct"] = pd.NA
    return TwinAdaptResult(trials, pd.DataFrame(window_records),
                           pd.DataFrame(transitions), pd.DataFrame(buffer_records),
                           pd.DataFrame(updates), model, len(pending))
