"""Historical compatibility shim.

Canonical implementation: src.detection.stage2.training_reference_hotelling.
"""
from src.detection.stage2.training_reference_hotelling import (
    TrainingReferenceHotellingConfig,
    calculate_training_reference_hotelling,
    validate_algorithm1_alarms,
)
__all__ = [
    "TrainingReferenceHotellingConfig",
    "calculate_training_reference_hotelling",
    "validate_algorithm1_alarms",
]
