from pathlib import Path

import numpy as np
import pytest

from src.bci_data import (
    dataset_2a_filename,
    load_bci_competition_iv_2a_session,
    load_cse_feature_file,
    resolve_dataset_2a_path,
)


def test_dataset_2a_filename_uses_official_naming() -> None:
    assert dataset_2a_filename(1, "T") == "A01T.gdf"
    assert dataset_2a_filename(9, "e") == "A09E.gdf"


def test_dataset_2a_filename_rejects_invalid_subject() -> None:
    with pytest.raises(ValueError, match="1 to 9"):
        dataset_2a_filename(10, "T")


def test_dataset_2a_filename_rejects_invalid_session() -> None:
    with pytest.raises(ValueError, match="'T' or 'E'"):
        dataset_2a_filename(1, "X")


def test_resolve_dataset_2a_path_finds_expected_file(
    tmp_path: Path,
) -> None:
    expected = tmp_path / "A03T.gdf"
    expected.write_bytes(b"placeholder")

    actual = resolve_dataset_2a_path(tmp_path, subject=3, session="T")

    assert actual == expected


def test_resolve_dataset_2a_path_reports_missing_file(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError, match="A02E.gdf"):
        resolve_dataset_2a_path(tmp_path, subject=2, session="E")


def test_load_cse_feature_file_returns_features_times_and_labels(
    tmp_path: Path,
) -> None:
    path = tmp_path / "subject_01_features.npz"
    features = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
    ])
    times = np.array([10, 11, 12])
    labels = np.array([1, 2, 1])

    np.savez(
        path,
        features=features,
        times=times,
        labels=labels,
    )

    loaded_features, loaded_times, loaded_labels = (
        load_cse_feature_file(path)
    )

    np.testing.assert_allclose(loaded_features, features)
    np.testing.assert_array_equal(loaded_times, times)
    np.testing.assert_array_equal(loaded_labels, labels)


def test_load_cse_feature_file_allows_missing_labels(
    tmp_path: Path,
) -> None:
    path = tmp_path / "unlabelled_features.npz"
    np.savez(
        path,
        features=np.ones((4, 3)),
        times=np.arange(4),
    )

    features, times, labels = load_cse_feature_file(path)

    assert features.shape == (4, 3)
    assert times.shape == (4,)
    assert labels is None


def test_load_cse_feature_file_rejects_length_mismatch(
    tmp_path: Path,
) -> None:
    path = tmp_path / "bad_features.npz"
    np.savez(
        path,
        features=np.ones((4, 2)),
        times=np.arange(3),
    )

    with pytest.raises(ValueError, match="equal length"):
        load_cse_feature_file(path)


def test_load_bci_session_uses_mne_and_returns_samples_by_channels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "A01T.gdf"
    path.write_bytes(b"placeholder")

    class FakeAnnotations:
        onset = np.array([1.0, 2.0])
        duration = np.array([0.0, 0.5])
        description = np.array(["769", "770"])

    class FakeRaw:
        def __init__(self) -> None:
            self.times = np.array([0.0, 0.004, 0.008])
            self.ch_names = ["EEG-C3", "EEG-C4"]
            self.info = {"sfreq": 250.0}
            self.annotations = FakeAnnotations()
            self.picked = False

        def copy(self) -> "FakeRaw":
            return self

        def pick(self, picks: str) -> "FakeRaw":
            assert picks == "eeg"
            self.picked = True
            return self

        def get_data(self) -> np.ndarray:
            return np.array([
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ])

    fake_raw = FakeRaw()

    class FakeIO:
        @staticmethod
        def read_raw_gdf(
            file_path: Path,
            *,
            preload: bool,
            verbose: str,
        ) -> FakeRaw:
            assert file_path == path
            assert preload is True
            assert verbose == "ERROR"
            return fake_raw

    class FakeMNE:
        io = FakeIO()

    monkeypatch.setattr(
        "src.bci_data._import_mne",
        lambda: FakeMNE(),
    )

    result = load_bci_competition_iv_2a_session(
        data_directory=tmp_path,
        subject=1,
        session="T",
    )

    assert fake_raw.picked is True
    assert result.subject == 1
    assert result.session == "T"
    assert result.signals.shape == (3, 2)
    assert result.n_samples == 3
    assert result.n_channels == 2
    assert result.channel_names == ("EEG-C3", "EEG-C4")
    assert result.sampling_frequency == 250.0
    assert result.annotations == (
        (1.0, 0.0, "769"),
        (2.0, 0.5, "770"),
    )
