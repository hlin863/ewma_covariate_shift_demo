"""Unit-level tests for sequential supervised adaptation semantics.

Scope:
- prediction occurs before any same-trial update
- confirmed shifts create adaptation events
- rejected/non-executable validation states do not create CSV-triggered updates
- classifier version changes only after retraining
- training-set size changes are auditable
- trial and adaptation traces expose required provenance fields

Use synthetic deterministic features here; full CSE execution belongs in
``tests/integration``.
"""
