"""Unit tests for the adaptive classifier lifecycle.

Scope:
- fit and fitted-state behaviour
- prediction and decision scores
- append-and-retrain behaviour
- retained training-set provenance
- classifier versioning
- classifier-specific input validation

Do not place EWMA, Stage-II, policy, or real-EEG tests in this module.
"""
