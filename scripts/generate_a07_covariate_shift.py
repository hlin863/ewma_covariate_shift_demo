"""Generate measured A07 μ/β CSP coordinates for the Flask visualisation."""

from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.bci.data import load_bci_competition_iv_2a_session
from src.bci.datasets.dataset2a import extract_dataset_2a_trials
from src.bci.datasets.dataset2a.evaluation_labels import load_dataset_2a_evaluation_labels
from src.bci.fbcsp import bandpass_trials, fit_binary_csp, log_normalised_variance
from src.web.covariate_shift import BAND_LABELS, PAPER_URL


def _boundary(points: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    classes = np.unique(labels)
    if classes.size != 2:
        raise ValueError("A classification boundary requires exactly two classes.")
    first = points[labels == classes[0]].mean(axis=0)
    second = points[labels == classes[1]].mean(axis=0)
    normal = second - first
    midpoint = (first + second) / 2.0
    return {"a": float(normal[0]), "b": float(normal[1]), "c": float(-normal @ midpoint)}


def _project_band(training, testing, low: float, high: float) -> tuple[np.ndarray, np.ndarray]:
    train_filtered = bandpass_trials(training.signals, training.sampling_frequency, low, high)
    test_filtered = bandpass_trials(testing.signals, testing.sampling_frequency, low, high)
    filters = fit_binary_csp(train_filtered, training.labels)[[0, -1]]
    train_projected = np.einsum("kc,tcs->tks", filters, train_filtered)
    test_projected = np.einsum("kc,tcs->tks", filters, test_filtered)
    return log_normalised_variance(train_projected), log_normalised_variance(test_projected)


def _points(values: np.ndarray) -> list[dict[str, float]]:
    return [{"x": float(row[0]), "y": float(row[1])} for row in values]


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--data-2a", type=Path, default=Path("data/raw/bci_competition_iv_2a"))
    parser.add_argument("--evaluation-labels", type=Path, required=True, help="Released A07E true-label MAT file")
    parser.add_argument("--output", type=Path, default=Path("outputs/visualisations/a07_covariate_shift.json"))
    args = parser.parse_args()

    train_session = load_bci_competition_iv_2a_session(args.data_2a, 7, "T")
    test_session = load_bci_competition_iv_2a_session(args.data_2a, 7, "E")
    released_labels = load_dataset_2a_evaluation_labels(args.evaluation_labels)
    training = extract_dataset_2a_trials(train_session)
    testing = extract_dataset_2a_trials(test_session, evaluation_labels=released_labels)

    bands = []
    for band_id, low, high in (("mu", 8.0, 12.0), ("beta", 14.0, 30.0)):
        train_values, test_values = _project_band(training, testing, low, high)
        bands.append({
            "id": band_id,
            **BAND_LABELS[band_id],
            "train": _points(train_values),
            "test": _points(test_values),
            "boundaries": {
                "train": _boundary(train_values, training.labels),
                "test": _boundary(test_values, testing.labels),
            },
        })

    payload = {
        "subject": "A07", "dataset": "BCI Competition IV Dataset 2A",
        "source": "computed_a07_gdf", "source_label": "Computed from A07 GDF sessions",
        "paper_url": PAPER_URL, "notice": "Computed locally from A07 Session I and II.", "bands": bands,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved A07 covariate-shift visualisation data to {args.output.resolve()}")


if __name__ == "__main__":
    main()
