"""Reference detectors used to compare the repository's EWMA methods."""

from src.detection.baselines.ici_cdt import (
    ICICDTConfig,
    ICICDTResult,
    ICICDTTrainingResult,
    fit_ici_cdt,
    run_ici_cdt,
)

__all__ = [
    "ICICDTConfig",
    "ICICDTResult",
    "ICICDTTrainingResult",
    "fit_ici_cdt",
    "run_ici_cdt",
]
