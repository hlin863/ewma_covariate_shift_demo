"""Contract tests between CSE outputs and the adaptation layer.

Scope:
- Stage-I/Stage-II result schemas reach adaptation unchanged
- ``confirmed_shift`` is interpreted consistently
- detector and adaptation trial times remain aligned
- confirmed/rejected validation outcomes produce the expected policy decisions
- adaptation-event counts are consistent with the selected policy

Use small deterministic detector outputs; real EEG data belong in
``tests/realdata``.
"""
