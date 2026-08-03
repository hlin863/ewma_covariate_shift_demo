"""Retired compatibility entry point.

The earlier implementation targeted a different paper's session-IV/session-V
percentage table and must not be used for the Dataset 2B Table 1 replication.
Use ``scripts/run_bci_2b_table1.py`` instead.
"""

raise SystemExit(
    "run_bci_2b_table3.py has been retired because it targeted the wrong "
    "published experiment. Run scripts/run_bci_2b_table1.py instead."
)
