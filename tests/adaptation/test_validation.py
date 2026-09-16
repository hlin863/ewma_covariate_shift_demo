"""Research-safeguard and invalid-input tests for adaptation experiments.

Scope:
- labelled-evaluation requirement for supervised adaptation
- feature/label/time alignment
- dimensionality consistency
- finite-value requirements
- duplicate or invalid trial identifiers
- safeguards against same-trial label leakage

These tests protect experimental validity rather than classifier accuracy.
"""
