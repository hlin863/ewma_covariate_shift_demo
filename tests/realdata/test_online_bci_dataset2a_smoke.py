"""Real-data smoke-test home for the Dataset 2A online-adaptation pipeline.

This module is intentionally separate from unit and integration tests because
it requires external BCI Competition IV Dataset 2A files. It should verify only
that one representative subject can traverse the complete feature -> CSE ->
adaptation path with valid output schemas.

Do not make exact published-result reproduction a smoke-test requirement.
"""
