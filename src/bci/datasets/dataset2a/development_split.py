"""Dataset 2A Session-I development split value object."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.bci_2a_experiment import Dataset2ATrialSignalResult


@dataclass(frozen=True)
class Dataset2ADevelopmentSplit:
    """Stratified Session-I split used for development and validation."""

    training: "Dataset2ATrialSignalResult"
    validation: "Dataset2ATrialSignalResult"
