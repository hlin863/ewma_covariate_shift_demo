"""Covariate-shift detection algorithms and preprocessing.

The canonical research-facing namespace groups PCA preprocessing, Stage-I EWMA
models, and Stage-II validation methods by function. Existing flat modules are
kept import-compatible while experiments migrate to this package.
"""

from src.detection.core import CSEConfig, CSEResult, run_cse
from src.detection.two_stage import (
    TSSDEWMAConfig,
    TSSDEWMAResult,
    run_tssd_ewma,
)
from src.detection.preprocessing import (
    CSEPCAResult,
    CSEReconstructionResult,
    extract_first_component,
    extract_retained_components,
    fit_cse_pca,
    reconstruct_cse_features,
    transform_cse_features,
)

__all__ = [
    "CSEConfig",
    "CSEResult",
    "CSEPCAResult",
    "CSEReconstructionResult",
    "TSSDEWMAConfig",
    "TSSDEWMAResult",
    "extract_first_component",
    "extract_retained_components",
    "fit_cse_pca",
    "reconstruct_cse_features",
    "run_cse",
    "run_tssd_ewma",
    "transform_cse_features",
]
