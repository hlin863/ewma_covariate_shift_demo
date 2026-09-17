"""BCI data loading and EEG feature-processing package.

This package is the canonical home for BCI Competition IV data access,
feature extraction, FBCSP processing, and dataset-specific experiment helpers.
Legacy flat ``src.*`` modules remain as compatibility shims while callers are
migrated to the structured namespace.
"""

from src.bci.data import (
    BCISessionData,
    dataset_2a_filename,
    dataset_2b_filename,
    load_bci_competition_iv_2a_session,
    load_bci_competition_iv_2b_session,
    load_cse_feature_file,
)
from src.bci.fbcsp import (
    CSE_UAEL_FILTER_BANK,
    ONLINE_BCI_2018_FILTER_BANK,
    CSPModel,
    FBCSPModel,
    PAPER_FILTER_BANK,
    bandpass_trials,
    fit_binary_csp,
    fit_fbcsp,
    fit_transform_fbcsp,
    log_normalised_variance,
    transform_fbcsp,
)

__all__ = [
    "BCISessionData",
    "CSE_UAEL_FILTER_BANK",
    "ONLINE_BCI_2018_FILTER_BANK",
    "CSPModel",
    "FBCSPModel",
    "PAPER_FILTER_BANK",
    "bandpass_trials",
    "dataset_2a_filename",
    "dataset_2b_filename",
    "fit_binary_csp",
    "fit_fbcsp",
    "fit_transform_fbcsp",
    "load_bci_competition_iv_2a_session",
    "load_bci_competition_iv_2b_session",
    "load_cse_feature_file",
    "log_normalised_variance",
    "transform_fbcsp",
]
