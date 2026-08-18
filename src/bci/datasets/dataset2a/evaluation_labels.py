from pathlib import Path

import numpy as np
from scipy.io import loadmat


def resolve_dataset_2a_evaluation_label_path(
    subject: int,
    *,
    data_directory: Path,
    labels_directory: Path | None = None,
) -> Path:
    """Resolve the released Session-II label file for one Dataset 2A subject.

    The official Dataset 2A signal archive contains GDF recordings.  The true
    labels for the evaluation sessions were released separately after the BCI
    Competition and therefore must not be assumed to live beside the GDF files.
    """

    if subject < 1 or subject > 9:
        raise ValueError("subject must be an integer from 1 to 9.")

    filename = f"A{subject:02d}E.mat"
    candidates: list[Path] = []

    if labels_directory is not None:
        candidates.append(Path(labels_directory) / filename)

    # Backward-compatible fallback for users who copied the released labels
    # into the same directory as the GDF files.
    candidates.append(Path(data_directory) / filename)

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    searched = "\n  - ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Dataset 2A Session-II true labels are not contained in the GDF signal "
        "archive. Download the official Dataset 2A evaluation labels released "
        "with the BCI Competition IV results, extract AxxE.mat files, and either "
        "place them in the Dataset 2A data directory or pass --labels-2a with "
        "their directory. Searched:\n  - " + searched
    )


def load_dataset_2a_evaluation_labels(
    label_path: Path,
) -> np.ndarray:
    path = Path(label_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset 2A evaluation-label file not found: {path}")

    data = loadmat(path)

    if "classlabel" not in data:
        raise ValueError(f"{path} does not contain 'classlabel'.")

    labels = np.asarray(data["classlabel"]).reshape(-1).astype(int)

    if labels.size != 288:
        raise ValueError(
            "Dataset 2A Session-II should contain "
            f"288 evaluation labels, got {labels.size}."
        )

    if not np.isin(labels, [1, 2, 3, 4]).all():
        raise ValueError("Dataset 2A labels must be in {1, 2, 3, 4}.")

    return labels
