"""Stage-II covariate-shift validation methods.

The separate Hotelling modules preserve distinct methodological interpretations.
"""
from src.detection.stage2.ks import (
    Stage2Config,
    Stage2ValidationResult,
    validate_stage_1_alarm,
    validate_stage_1_alarms,
)
from src.detection.stage2.adaptive_window_hotelling import (
    AdaptiveHotellingConfig,
    validate_adaptive_window_alarms,
)
from src.detection.stage2.window_feasibility import (
    WindowSelection,
    WindowSelectionPolicy,
    minimum_equal_window_size,
    select_validation_window,
)
from src.detection.stage2.paper_two_sample_hotelling import (
    PaperTwoSampleHotellingConfig,
    calculate_paper_two_sample_hotelling,
    validate_paper_two_sample_alarms,
)
from src.detection.stage2.retrospective_hotelling import (
    HotellingConfig,
    HotellingValidationResult,
    calculate_hotelling_t_squared,
    validate_multivariate_alarm,
    validate_multivariate_alarms,
)
from src.detection.stage2.training_reference_hotelling import (
    TrainingReferenceHotellingConfig,
    calculate_training_reference_hotelling,
    validate_algorithm1_alarms,
)
__all__ = [
    "AdaptiveHotellingConfig",
    "WindowSelection",
    "WindowSelectionPolicy",
    "minimum_equal_window_size",
    "select_validation_window",
    "validate_adaptive_window_alarms",
    "HotellingConfig",
    "HotellingValidationResult",
    "PaperTwoSampleHotellingConfig",
    "Stage2Config",
    "Stage2ValidationResult",
    "TrainingReferenceHotellingConfig",
    "calculate_hotelling_t_squared",
    "calculate_paper_two_sample_hotelling",
    "calculate_training_reference_hotelling",
    "validate_algorithm1_alarms",
    "validate_multivariate_alarm",
    "validate_multivariate_alarms",
    "validate_paper_two_sample_alarms",
    "validate_stage_1_alarm",
    "validate_stage_1_alarms",
]
