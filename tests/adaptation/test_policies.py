"""Unit tests for adaptation-policy decision rules.

Scope:
- never-update policy
- periodic-update policy
- Stage-I warning-triggered policy
- Stage-II validated-shift-triggered policy
- treatment of rejected, pending, skipped, and ineligible validation states

Keep classifier fitting and detector execution out of this module.
"""
