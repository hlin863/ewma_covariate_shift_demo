from argparse import ArgumentTypeError

import pytest

from scripts.run_bci_table1_reproduction import _parse_pca_components


def test_parse_pca_components_accepts_integer_count() -> None:
    assert _parse_pca_components("1") == 1
    assert _parse_pca_components("3") == 3


def test_parse_pca_components_accepts_variance_fraction() -> None:
    assert _parse_pca_components("0.95") == 0.95


def test_parse_pca_components_accepts_all() -> None:
    assert _parse_pca_components("all") is None
    assert _parse_pca_components("none") is None


@pytest.mark.parametrize("value", ["0", "-1", "1.0", "abc"])
def test_parse_pca_components_rejects_invalid_values(value: str) -> None:
    with pytest.raises(ArgumentTypeError):
        _parse_pca_components(value)
