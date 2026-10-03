"""Historical compatibility shim.

Canonical implementation: src.detection.stage2.retrospective_hotelling.
"""
from src.detection.stage2.retrospective_hotelling import (
    HotellingConfig,
    HotellingValidationResult,
    calculate_hotelling_t_squared,
    validate_multivariate_alarm,
    validate_multivariate_alarms,
)
__all__ = [
    "HotellingConfig",
    "HotellingValidationResult",
    "calculate_hotelling_t_squared",
    "validate_multivariate_alarm",
    "validate_multivariate_alarms",
]
