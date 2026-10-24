"""Companion checks for abcd_annotated.py (run: pytest -q test_abcd_annotated.py)."""

from pathlib import Path
import importlib.util
import math
import numpy as np
import pytest

MODULE_PATH = Path(__file__).with_name('abcd_annotated.py')
spec = importlib.util.spec_from_file_location('abcd_annotated', MODULE_PATH)
abcd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(abcd)


def test_equation_5_matches_direct_substitution():
    n1, n2, v1, v2, eps, M, kappa = 20, 20, .01, .01, .8, 1., .5
    e1, e2 = kappa * eps, (1 - kappa) * eps
    expected = (2 * math.exp(-n1 * e1 * e1 / (2 * (v1 + M * e1 / 3)))
                + 2 * math.exp(-n2 * e2 * e2 / (2 * (v2 + M * e2 / 3))))
    observed = abcd.bernstein_two_sample_upper_bound(n1, n2, v1, v2, eps, M, kappa)
    assert math.isclose(observed, expected, rel_tol=1e-14)
    assert math.isclose(observed, 0.0000567703522150656, rel_tol=1e-12)


def test_zero_epsilon_has_raw_upper_bound_four():
    assert abcd.bernstein_two_sample_upper_bound(20, 20, .01, .01, 0., 1., .5) == 4.


def test_equation_8_selects_split_after_four():
    losses = [.02, .03, .02, .03, .12, .13, .11, .14]
    result = abcd.scan_candidate_splits(losses)
    assert result['split_after'] == 4
    assert math.isclose(result['epsilon'], .1, abs_tol=1e-14)
    assert math.isclose(result['kappa'], .5, abs_tol=1e-14)
    assert math.isclose(result['upper_bound_score'], 2.9685677986483334, rel_tol=1e-12)
    assert result['illustrative_flag'] is False


def test_constant_losses_cause_no_illustrative_alarm():
    result = abcd.scan_candidate_splits(np.full(12, .02))
    assert result['epsilon'] == pytest.approx(0.)
    assert result['upper_bound_score'] == pytest.approx(4.)
    assert result['illustrative_flag'] is False


@pytest.mark.parametrize('losses', [[], [0., 1., 2.], [0., 1., float('nan'), 2.]])
def test_scan_rejects_short_or_nonfinite_sequences(losses):
    with pytest.raises(ValueError):
        abcd.scan_candidate_splits(losses)


@pytest.mark.parametrize('kwargs', [
    dict(n1=1, n2=3, v1=0, v2=0, eps=.1, M=1, kappa=.5),
    dict(n1=3, n2=3, v1=-.1, v2=0, eps=.1, M=1, kappa=.5),
    dict(n1=3, n2=3, v1=0, v2=0, eps=.1, M=1, kappa=0),
    dict(n1=3, n2=3, v1=0, v2=0, eps=-.1, M=1, kappa=.5),
])
def test_upper_bound_rejects_invalid_inputs(kwargs):
    with pytest.raises(ValueError):
        abcd.bernstein_two_sample_upper_bound(**kwargs)
