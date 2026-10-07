"""Load NASA C-MAPSS trajectories without fitting preprocessing or models."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

COLUMN_NAMES = [
    "unit",
    "cycle",
    "setting_1",
    "setting_2",
    "setting_3",
    *[f"sensor_{i}" for i in range(1, 22)],
]

_VALID_DATASETS = {"FD001", "FD002", "FD003", "FD004"}
_DEFAULT_DIRECTORY = (
    Path(__file__).resolve().parents[4] / "data" / "raw" / "turbofan_engine_degradation"
)


@dataclass(frozen=True)
class TurbofanDataset:
    """Raw tables sorted by unit/cycle, with one endpoint RUL per test unit.

    ``test_rul[i]`` belongs to test unit ``i + 1``. Train and test unit
    numbers identify separate fleets and must not be joined across splits.
    RUL helpers return new tables; the 26-column raw tables stay unchanged.
    Test targets are offline evaluation/oracle information, not features.
    """

    training: pd.DataFrame
    testing: pd.DataFrame
    test_rul: np.ndarray
    dataset: str

    @classmethod
    def load(
        cls,
        data_directory: str | Path | None = None,
        dataset: str = "FD001",
    ) -> "TurbofanDataset":
        """Read train, test and RUL files for one FD001–FD004 subset.

        The default directory is resolved from this module, so loading
        does not depend on the caller's current working directory.
        """
        subset = str(dataset).strip().upper()
        if subset not in _VALID_DATASETS:
            raise ValueError("dataset must be FD001, FD002, FD003 or FD004.")
        directory = _DEFAULT_DIRECTORY if data_directory is None else Path(data_directory)

        training = _read_trajectories(directory / f"train_{subset}.txt")
        testing = _read_trajectories(directory / f"test_{subset}.txt")
        rul_path = directory / f"RUL_{subset}.txt"
        rul_values = _read_numeric_table(rul_path, n_columns=1)[:, 0]
        if (rul_values < 0).any() or (rul_values != np.floor(rul_values)).any():
            raise ValueError(f"{rul_path}: RUL values must be non-negative integer cycles.")
        if rul_values.size != testing["unit"].nunique():
            raise ValueError(f"{rul_path}: expected one RUL value per test unit.")

        return cls(
            training=training,
            testing=testing,
            test_rul=rul_values,
            dataset=subset,
        )

    def training_with_rul(self) -> pd.DataFrame:
        """Return uncapped RUL = final training cycle minus current cycle."""
        final_cycles = self.training.groupby("unit")["cycle"].transform("max")
        return self.training.assign(rul=final_cycles - self.training["cycle"])

    def testing_with_rul(self) -> pd.DataFrame:
        """Return offline test RUL, including the supplied endpoint offset.

        Test trajectories stop before failure. Their final cycle must have
        the NASA-supplied RUL, rather than an incorrectly assumed zero.
        """
        units = np.sort(self.testing["unit"].unique())
        endpoint_rul = pd.Series(self.test_rul, index=units)
        final_cycles = self.testing.groupby("unit")["cycle"].transform("max")
        return self.testing.assign(
            rul=final_cycles - self.testing["cycle"] + self.testing["unit"].map(endpoint_rul)
        )


def _read_numeric_table(file_path: Path, *, n_columns: int) -> np.ndarray:
    if not file_path.is_file():
        raise FileNotFoundError(f"C-MAPSS file was not found: {file_path}")
    with file_path.open(encoding="utf-8") as source:
        if not any(line.strip() for line in source):
            raise ValueError(f"{file_path}: file must contain at least one observation.")
    try:
        values = np.loadtxt(file_path, dtype=float, ndmin=2, comments=None)
    except ValueError as exc:
        raise ValueError(f"{file_path}: expected a rectangular numeric table.") from exc
    if values.shape[1] != n_columns:
        raise ValueError(f"{file_path}: expected {n_columns} columns, got {values.shape[1]}.")
    if not np.isfinite(values).all():
        raise ValueError(f"{file_path}: values must be finite.")
    return values


def _read_trajectories(file_path: Path) -> pd.DataFrame:
    values = _read_numeric_table(file_path, n_columns=len(COLUMN_NAMES))
    identifiers = values[:, :2]
    if (
        (identifiers < 1).any()
        or (identifiers != np.floor(identifiers)).any()
        or (identifiers >= np.iinfo(np.int64).max).any()
    ):
        raise ValueError(f"{file_path}: unit and cycle must be positive integer identifiers.")
    frame = pd.DataFrame(values, columns=COLUMN_NAMES)
    frame[["unit", "cycle"]] = frame[["unit", "cycle"]].astype(np.int64)
    frame = frame.sort_values(["unit", "cycle"]).reset_index(drop=True)
    units = frame["unit"].unique()
    if not np.array_equal(units, np.arange(1, units.size + 1)):
        raise ValueError(f"{file_path}: unit IDs must be consecutive starting at 1.")
    expected_cycles = frame.groupby("unit").cumcount().to_numpy() + 1
    if not np.array_equal(frame["cycle"].to_numpy(), expected_cycles):
        raise ValueError(f"{file_path}: each unit must have unique consecutive cycles starting at 1.")
    return frame


def load_turbofan_dataset(
    data_directory: str | Path | None = None,
    dataset: str = "FD001",
) -> TurbofanDataset:
    """Load one C-MAPSS subset through the repository's function-style API."""
    return TurbofanDataset.load(data_directory, dataset)
