"""Data contract for the A07 train/test covariate-shift visualisation."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any


PAPER_URL = "https://pmc.ncbi.nlm.nih.gov/articles/PMC7086459/"
BAND_LABELS = {
    "mu": {"title": "μ band", "range": "8–12 Hz"},
    "beta": {"title": "β band", "range": "14–30 Hz"},
}


def _cloud(
    centre_x: float,
    centre_y: float,
    spread_x: float,
    spread_y: float,
    angle: float,
    *,
    seed: int,
    count: int = 54,
) -> list[dict[str, float | int]]:
    """Create a deterministic paper-style cloud, never presented as measured data."""

    rng = random.Random(seed)
    cosine, sine = math.cos(angle), math.sin(angle)
    points: list[dict[str, float | int]] = []
    for _ in range(count):
        local_x = rng.gauss(0.0, spread_x)
        local_y = rng.gauss(0.0, spread_y)
        x = centre_x + local_x * cosine - local_y * sine
        y = centre_y + local_x * sine + local_y * cosine
        points.append({"x": round(x, 4), "y": round(y, 4)})
    return points


def paper_schematic() -> dict[str, Any]:
    """Return an immediate visual explanation matching the paper's Fig. 1 semantics."""

    return {
        "subject": "A07",
        "dataset": "BCI Competition IV Dataset 2A",
        "source": "paper_schematic",
        "source_label": "Paper-aligned schematic",
        "paper_url": PAPER_URL,
        "notice": (
            "This view reconstructs the relationships described in Fig. 1; points are "
            "deterministic illustrative coordinates, not digitised or measured A07 samples."
        ),
        "bands": [
            {
                "id": "mu",
                **BAND_LABELS["mu"],
                "train": _cloud(-0.55, 0.35, 0.52, 0.23, 0.42, seed=701),
                "test": _cloud(0.48, -0.23, 0.50, 0.22, 0.32, seed=702),
                "boundaries": {
                    "train": {"a": 0.72, "b": -1.0, "c": 0.18},
                    "test": {"a": 0.72, "b": -1.0, "c": -0.38},
                },
            },
            {
                "id": "beta",
                **BAND_LABELS["beta"],
                "train": _cloud(-0.42, -0.18, 0.43, 0.28, -0.48, seed=703),
                "test": _cloud(0.58, 0.34, 0.47, 0.30, -0.38, seed=704),
                "boundaries": {
                    "train": {"a": -0.58, "b": -1.0, "c": -0.12},
                    "test": {"a": -0.58, "b": -1.0, "c": 0.43},
                },
            },
        ],
    }


def load_visualisation(path: str | Path) -> dict[str, Any] | None:
    visualisation_path = Path(path)
    if not visualisation_path.is_file():
        return None
    payload = json.loads(visualisation_path.read_text(encoding="utf-8"))
    if payload.get("subject") != "A07":
        raise ValueError("A07 visualisation data must identify subject 'A07'.")
    bands = payload.get("bands")
    if not isinstance(bands, list) or {band.get("id") for band in bands} != {"mu", "beta"}:
        raise ValueError("A07 visualisation data must contain mu and beta bands.")
    for band in bands:
        if not isinstance(band.get("train"), list) or not isinstance(band.get("test"), list):
            raise ValueError("Each band must contain train and test point arrays.")
    return payload


def visualisation_for(path: str | Path) -> tuple[dict[str, Any], bool]:
    measured = load_visualisation(path)
    return (measured, True) if measured is not None else (paper_schematic(), False)
