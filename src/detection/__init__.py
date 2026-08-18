"""Covariate-shift detection algorithms and preprocessing.

The canonical research-facing namespace groups PCA preprocessing, Stage-I EWMA
models, and Stage-II validation methods by function. Existing flat modules are
kept import-compatible while experiments migrate to this package.
"""

from src.cse import CSEConfig, CSEResult, run_cse
from src.detection.preprocessing import (
    CSEPCAResult,
    extract_first_component,
    extract_retained_components,
    fit_cse_pca,
    transform_cse_features,
)

__all__ = [
    "CSEConfig",
    "CSEResult",
    "CSEPCAResult",
    "extract_first_component",
    "extract_retained_components",
    "fit_cse_pca",
    "run_cse",
    "transform_cse_features",
]
