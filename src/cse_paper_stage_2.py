"""Historical compatibility shim.

Canonical implementation: src.detection.stage2.paper_two_sample_hotelling.
"""
from src.detection.stage2.paper_two_sample_hotelling import (
    PaperTwoSampleHotellingConfig,
    calculate_paper_two_sample_hotelling,
    validate_paper_two_sample_alarms,
)
__all__ = [
    "PaperTwoSampleHotellingConfig",
    "calculate_paper_two_sample_hotelling",
    "validate_paper_two_sample_alarms",
]
