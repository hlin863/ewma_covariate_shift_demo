"""The 2019 session protocol preserves trial boundaries and feature fit scope."""

import numpy as np
import pytest

from src.bci.datasets.dataset2b import (
    TrialSignalResult,
    build_dataset_2b_development_fbcsp_features,
    prepare_cse_uael_2019_protocol,
)
from src.bci.fbcsp import fit_fbcsp


def _sessions() -> list[TrialSignalResult]:
    rng = np.random.default_rng(101)
    sessions = []
    sfreq = 100.0
    t = np.arange(300) / sfreq
    for number in range(1, 6):
        labels = np.tile([0, 1], 10) if number <= 3 else np.full(8, -1)
        signals = rng.normal(scale=0.2, size=(len(labels), 3, 300))
        for index, label in enumerate(labels):
            if label >= 0:
                signals[index, 0 if label == 0 else 2] += 1.2 * np.sin(2 * np.pi * 10 * t)
        session_id = f"{number:02d}{'T' if number <= 3 else 'E'}"
        sessions.append(TrialSignalResult(
            signals=signals, labels=labels, times=np.arange(len(labels)),
            session_ids=np.full(len(labels), session_id),
            cue_descriptions=np.full(len(labels), "783" if number > 3 else "769"),
            channel_names=("C3", "Cz", "C4"), sampling_frequency=sfreq,
        ))
    return sessions


def test_2019_protocol_is_stratified_reproducible_and_keeps_evaluation_out() -> None:
    sessions = _sessions()
    protocol = prepare_cse_uael_2019_protocol(sessions)
    second = prepare_cse_uael_2019_protocol(sessions)
    training = protocol.development.training
    validation = protocol.development.validation
    assert training.signals.shape[0] == 42
    assert validation.signals.shape[0] == 18
    np.testing.assert_array_equal(np.bincount(training.labels), [21, 21])
    np.testing.assert_array_equal(np.bincount(validation.labels), [9, 9])
    np.testing.assert_array_equal(training.signals, second.development.training.signals)
    assert np.isin(training.session_ids, ["01T", "02T", "03T"]).all()
    assert np.isin(protocol.evaluation.session_ids, ["04E", "05E"]).all()
    assert protocol.evaluation.signals.shape[0] == 16
    # The pool's unique times survive selection, so no observation is lost or duplicated.
    assert set(training.times).isdisjoint(validation.times)
    assert set(training.times) | set(validation.times) == set(range(60))


def test_2019_protocol_rejects_wrong_order_and_unlabelled_training() -> None:
    sessions = _sessions()
    with pytest.raises(ValueError, match="session 1"):
        prepare_cse_uael_2019_protocol(sessions[::-1])
    sessions[0] = TrialSignalResult(
        **{**vars(sessions[0]), "labels": np.full(20, -1)}
    )
    with pytest.raises(ValueError, match="labelled classes"):
        prepare_cse_uael_2019_protocol(sessions)


def test_2b_fbcsp_fits_training_only_and_transforms_validation() -> None:
    protocol = prepare_cse_uael_2019_protocol(_sessions())
    pipeline = build_dataset_2b_development_fbcsp_features(
        protocol.development, protocol.evaluation
    )
    expected = fit_fbcsp(
        protocol.development.training.signals,
        protocol.development.training.labels,
        protocol.development.training.sampling_frequency,
    )
    for actual, reference in zip(pipeline.model.models, expected.models, strict=True):
        np.testing.assert_allclose(actual.filters, reference.filters)
    assert pipeline.training.features.shape == (42, 20)
    assert pipeline.validation.features.shape == (18, 20)
    assert pipeline.evaluation.features.shape == (16, 20)
    assert pipeline.testing is pipeline.evaluation
