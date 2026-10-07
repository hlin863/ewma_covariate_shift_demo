import pandas as pd
from dataclasses import dataclass
import numpy as np

COLUMN_NAMES = [
    "unit",
    "cycle",
    "setting_1",
    "setting_2",
    "setting_3",
    *[f"sensor_{i}" for i in range(1, 22)],
]

@dataclass(frozen=True)
class TurbofanDataset:
    training: pd.DataFrame
    testing: pd.DataFrame
    test_rul: np.ndarray
    dataset: str

    