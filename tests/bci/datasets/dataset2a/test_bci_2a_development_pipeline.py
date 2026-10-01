import numpy as np

from src.bci_2a_development_pipeline import (
    build_dataset_2a_development_fbcsp_features,
)
from src.bci_2a_experiment import (
    DATASET_2A_CHANNELS,
    Dataset2ATrialSignalResult,
    split_dataset_2a_session1,
)
from src.fbcsp import fit_fbcsp


def _trials(session_id: str, n_trials: int, *, labelled: bool) -> Dataset2ATrialSignalResult:
    rng = np.random.default_rng(120 if session_id == "T" else 121)
    sfreq = 100.0
    n_samples = 300
    signals = rng.normal(scale=0.2, size=(n_trials, 10, n_samples))
    labels = np.asarray([index % 2 for index in range(n_trials)], dtype=int)
    time = np.arange(n_samples) / sfreq
    for index, label in enumerate(labels):
        channel = 0 if label == 0 else 5
        signals[index, channel] += 1.2 * np.sin(2 * np.pi * 10.0 * time)
    if not labelled:
        labels = np.full(n_trials, -1, dtype=int)
    return Dataset2ATrialSignalResult(
        signals=signals,
        labels=labels,
        times=np.arange(n_trials, dtype=int),
        cue_descriptions=np.asarray(
            ["769" if label == 0 else "770" for label in labels]
            if labelled
            else ["783"] * n_trials,
            dtype="U8",
        ),
        channel_names=DATASET_2A_CHANNELS,
        sampling_frequency=sfreq,
        session_id=session_id,
    )


def test_development_pipeline_uses_70_percent_only_for_fbcsp_fit() -> None:
    session1 = _trials("T", 20, labelled=True)
    session2 = _trials("E", 8, labelled=False)
    split = split_dataset_2a_session1(
        session1,
        validation_fraction=0.30,
        random_state=42,
    )

    pipeline = build_dataset_2a_development_fbcsp_features(split, session2)
    direct_training_only_model = fit_fbcsp(
        split.training.signals,
        split.training.labels,
        split.training.sampling_frequency,
    )

    assert pipeline.training.features.shape == (14, 20)
    assert pipeline.validation.features.shape == (6, 20)
    assert pipeline.testing.features.shape == (8, 20)
    for actual, expected in zip(
        pipeline.model.models,
        direct_training_only_model.models,
        strict=True,
    ):
        np.testing.assert_allclose(actual.filters, expected.filters)


def test_development_pipeline_transforms_validation_and_test_without_labels() -> None:
    session1 = _trials("T", 20, labelled=True)
    session2 = _trials("E", 8, labelled=False)
    split = split_dataset_2a_session1(session1, random_state=7)
    pipeline = build_dataset_2a_development_fbcsp_features(split, session2)

    assert pipeline.validation.session_id == "T"
    assert pipeline.testing.session_id == "E"
    assert len(pipeline.training.feature_names) == 20
    assert pipeline.training.feature_names == pipeline.validation.feature_names
    assert pipeline.training.feature_names == pipeline.testing.feature_names
    assert np.isfinite(pipeline.training.features).all()
    assert np.isfinite(pipeline.validation.features).all()
    assert np.isfinite(pipeline.testing.features).all()
