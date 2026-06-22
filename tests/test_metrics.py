"""Behavior tests for the concordance index."""

import numpy as np

from src.metrics import concordance_index


def test_perfect_concordance():
    # Risk perfectly ranks earlier failures higher.
    time = np.array([1.0, 2.0, 3.0, 4.0])
    event = np.array([1.0, 1.0, 1.0, 1.0])
    risk = np.array([4.0, 3.0, 2.0, 1.0])
    assert concordance_index(time, event, risk) == 1.0


def test_worst_concordance():
    time = np.array([1.0, 2.0, 3.0, 4.0])
    event = np.array([1.0, 1.0, 1.0, 1.0])
    risk = np.array([1.0, 2.0, 3.0, 4.0])  # exactly reversed
    assert concordance_index(time, event, risk) == 0.0


def test_tied_risk_is_half():
    time = np.array([1.0, 2.0])
    event = np.array([1.0, 1.0])
    risk = np.array([0.5, 0.5])
    assert concordance_index(time, event, risk) == 0.5


def test_no_comparable_pairs_returns_half():
    # All censored: no comparable pair exists.
    time = np.array([1.0, 2.0, 3.0])
    event = np.array([0.0, 0.0, 0.0])
    risk = np.array([3.0, 2.0, 1.0])
    assert concordance_index(time, event, risk) == 0.5


def test_censoring_handled():
    # Subject 1 is censored at time 2, so the pair (0,1) is comparable only
    # from 0's side (0 fails at 1 < 2). Subject 0 has higher risk: concordant.
    time = np.array([1.0, 2.0])
    event = np.array([1.0, 0.0])
    risk = np.array([2.0, 1.0])
    assert concordance_index(time, event, risk) == 1.0
