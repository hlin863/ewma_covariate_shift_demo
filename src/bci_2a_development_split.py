"""Dataset 2A Session-I development split for paper-oriented experiments.

The 2019 CSE-UAEL paper states that the existing training dataset was further
partitioned into 70% training and 30% validation subsets.  This module keeps
that development split as an explicit value object so the Dataset 2A
experiment can distinguish model-fitting observations from parameter-selection
observations without mixing either subset with Session-II evaluation data.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.bci_2a_experiment import Dataset2ATrialSignalResult


@dataclass(frozen=True)
class Dataset2ADevelopmentSplit:
    """Stratified Session-I split used for development and validation.

    ``training`` is the subset used to fit data-dependent transformations and
    CSE training state. ``validation`` is held out from those fitting steps and
    is reserved for parameter-selection experiments.
    """

    training: "Dataset2ATrialSignalResult"
    validation: "Dataset2ATrialSignalResult"
