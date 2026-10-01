# BCI coverage

`test_bci_data.py` and `test_filter_bank_variants.py` cover shared loading and
FBCSP contracts. `datasets/dataset2a`, `datasets/dataset2b`, and
`datasets/chowdhury` hold dataset-specific protocol, feature and metadata cases.
`pipelines/test_cse_bci_features.py` checks exported features through CSE.

These cases use synthetic signals, temporary files, mocked loading, and the
committed cohort metadata. They do not constitute a raw-EEG reproduction run.
