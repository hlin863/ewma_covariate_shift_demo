from pathlib import Path
import numpy as np
from scipy.io import loadmat


def load_dataset_2a_evaluation_labels(
    label_path: Path,
) -> np.ndarray:
    data = loadmat(label_path)

    if "classlabel" not in data:
        raise ValueError(f"{label_path} does not contain 'classlabel'.")

    labels = np.asarray(data["classlabel"]).reshape(-1).astype(int)

    if labels.size != 288:
        raise ValueError(
            "Dataset 2A Session-II should contain "
            f"288 evaluation labels, got {labels.size}."
        )

    if not np.isin(labels, [1, 2, 3, 4]).all():
        raise ValueError("Dataset 2A labels must be in {1, 2, 3, 4}.")

    return labels
