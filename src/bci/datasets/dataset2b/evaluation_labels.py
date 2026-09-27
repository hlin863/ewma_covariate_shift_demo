"""Official evaluation-label helpers for BCI Competition IV Dataset 2B."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import loadmat


_EVALUATION_SESSIONS = {4: "E", 5: "E"}


def dataset_2b_evaluation_label_filename(subject: int, session: int) -> str:
    """Return the official MAT label filename for Dataset 2B session IV or V."""

    if subject < 1 or subject > 9:
        raise ValueError("subject must be an integer from 1 to 9.")
    if session not in _EVALUATION_SESSIONS:
        raise ValueError("Dataset 2B evaluation session must be 4 or 5.")
    return f"B{subject:02d}{session:02d}E.mat"


def resolve_dataset_2b_evaluation_label_path(
    subject: int,
    session: int,
    *,
    data_directory: str | Path,
    labels_directory: str | Path | None = None,
) -> Path:
    """Resolve a released Dataset 2B evaluation-label MAT file.

    Dataset 2B evaluation GDF files contain unknown-cue annotations (783).
    Accuracy on sessions IV/V therefore requires the separately released
    classlabel MAT files. The helper first checks an explicit labels directory
    and then the signal directory for users who extracted both archives together.
    """

    filename = dataset_2b_evaluation_label_filename(subject, session)
    candidates: list[Path] = []
    if labels_directory is not None:
        candidates.append(Path(labels_directory) / filename)
    candidates.append(Path(data_directory) / filename)

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    searched = "\n  - ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Dataset 2B sessions IV/V require the official released evaluation "
        "labels to calculate classification accuracy. Expected a MAT file "
        f"named {filename}. Searched:\n  - {searched}"
    )


def load_dataset_2b_evaluation_labels(label_path: str | Path) -> np.ndarray:
    """Load one official Dataset 2B evaluation label vector as 0/1 classes."""

    path = Path(label_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset 2B evaluation-label file not found: {path}")

    data = loadmat(path)
    if "classlabel" not in data:
        raise ValueError(f"{path} does not contain 'classlabel'.")

    labels = np.asarray(data["classlabel"]).reshape(-1).astype(int)
    if labels.size != 160:
        raise ValueError(
            "Dataset 2B evaluation sessions IV/V should contain 160 labels; "
            f"received {labels.size}."
        )

    unique = set(np.unique(labels).tolist())
    if unique.issubset({1, 2}):
        return labels - 1
    if unique.issubset({0, 1}):
        return labels

    raise ValueError(
        "Dataset 2B evaluation labels must encode the two motor-imagery "
        "classes as {1, 2} or {0, 1}."
    )
