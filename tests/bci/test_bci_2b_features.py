from pathlib import Path

import numpy as np
import pytest

from src.bci_data import (
    BCISessionData,
    dataset_2b_filename,
    resolve_dataset_2b_path,
)
from src.bci_features import (
    concatenate_feature_results,
    extract_bandpower_windows,
)


def _session(
    signals: np.ndarray,
    *,
    session_name: str = "01T",
    sampling_frequency: float = 100.0,
) -> BCISessionData:
    times = np.arange(signals.shape[0]) / sampling_frequency
    return BCISessionData(
        subject=1,
        session=session_name,
        file_path=Path("dummy.gdf"),
        signals=signals,
        times=times,
        channel_names=("C3", "Cz", "C4"),
        sampling_frequency=sampling_frequency,
        annotations=(),
    )


def test_dataset_2b_filenames_follow_session_suffixes() -> None:
    assert dataset_2b_filename(1, 1) == "B0101T.gdf"
    assert dataset_2b_filename(1, 3) == "B0103T.gdf"
    assert dataset_2b_filename(1, 4) == "B0104E.gdf"
    assert dataset_2b_filename(9, 5) == "B0905E.gdf"


@pytest.mark.parametrize("session", [0, 6])
def test_dataset_2b_filename_rejects_invalid_session(session: int) -> None:
    with pytest.raises(ValueError, match="integer from 1 to 5"):
        dataset_2b_filename(1, session)


def test_resolve_dataset_2b_path(tmp_path: Path) -> None:
    expected = tmp_path / "B0101T.gdf"
    expected.write_bytes(b"placeholder")
    assert resolve_dataset_2b_path(tmp_path, 1, 1) == expected


def test_bandpower_extraction_returns_channel_band_features() -> None:
    sampling_frequency = 100.0
    times = np.arange(600) / sampling_frequency
    signals = np.column_stack(
        [
            np.sin(2 * np.pi * 10.0 * times),
            np.sin(2 * np.pi * 20.0 * times),
            np.sin(2 * np.pi * 10.0 * times)
            + 0.5 * np.sin(2 * np.pi * 20.0 * times),
        ]
    )

    result = extract_bandpower_windows(
        _session(signals, sampling_frequency=sampling_frequency),
        window_seconds=2.0,
        step_seconds=1.0,
    )

    assert result.features.shape == (5, 6)
    assert result.times.shape == (5,)
    assert result.feature_names == (
        "C3_mu",
        "C3_beta",
        "Cz_mu",
        "Cz_beta",
        "C4_mu",
        "C4_beta",
    )
    assert np.isfinite(result.features).all()


def test_bandpower_extraction_skips_nan_separator_windows() -> None:
    signals = np.ones((600, 3))
    signals[250:350] = np.nan

    result = extract_bandpower_windows(
        _session(signals),
        window_seconds=1.0,
        step_seconds=1.0,
    )

    assert result.features.shape[0] == 4
    assert np.isfinite(result.features).all()


def test_concatenate_feature_results_creates_unique_cse_times() -> None:
    first = extract_bandpower_windows(
        _session(np.ones((300, 3)), session_name="01T"),
        window_seconds=1.0,
        step_seconds=1.0,
    )
    second = extract_bandpower_windows(
        _session(np.ones((300, 3)) * 2.0, session_name="02T"),
        window_seconds=1.0,
        step_seconds=1.0,
    )

    combined = concatenate_feature_results([first, second])

    assert combined.features.shape[0] == 6
    np.testing.assert_array_equal(combined.times, np.arange(6))
    assert combined.session_ids.tolist() == [
        "01T",
        "01T",
        "01T",
        "02T",
        "02T",
        "02T",
    ]
