"""End-to-end synthetic test for the 2018-style online BCI adaptation flow.

Scope:
- calibration classifier creation
- sequential evaluation
- Stage-I warning and Stage-II confirmation
- validated-shift-triggered retraining
- classifier-version transition
- subsequent prediction with the updated classifier

This is a computational flow test, not a reproduction of the original
clinical participant experiment.
"""
