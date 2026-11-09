"""Serializable results of the TWIN-ADAPT BCI experiment."""
from dataclasses import dataclass
import pandas as pd


@dataclass
class TwinAdaptResult:
    trials: pd.DataFrame
    accuracy_windows: pd.DataFrame
    transitions: pd.DataFrame
    adaptive_windows: pd.DataFrame
    updates: pd.DataFrame
    final_classifier: object
    pending_feedback_batches: int

    @property
    def accuracy(self) -> float:
        return float(self.trials["correct"].mean()) if not self.trials.empty else float("nan")

    @property
    def update_count(self) -> int:
        return len(self.updates)
