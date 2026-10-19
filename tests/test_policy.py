from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from churnpolicy.baseline import heuristic_flag, heuristic_score
from churnpolicy.config import PolicyParams
from churnpolicy.evaluate import flag_top_fraction
from churnpolicy.policy import expected_value, outcome, recommend

PARAMS = PolicyParams(capacity=3, save_rate=0.3, contact_cost=40, value_horizon_months=12)


def test_recommend_respects_capacity():
    rng = np.random.default_rng(0)
    p, rev = rng.random(1000), rng.lognormal(4, 1, 1000)
    for k in (0, 1, 50, 2000):
        sel = recommend(p, rev, PolicyParams(k, 0.3, 40, 12))
        assert sel.sum() <= k


def test_recommend_is_optimal_against_brute_force():
    rng = np.random.default_rng(1)
    for _ in range(25):
        n = 9
        p, rev = rng.random(n) * 0.5, rng.lognormal(3.5, 1.2, n)
        ev = expected_value(p, rev, PARAMS)
        best = max(
            (sum(ev[list(c)]) for r in range(PARAMS.capacity + 1) for c in itertools.combinations(range(n), r)),
            default=0.0,
        )
        assert np.isclose(ev[recommend(p, rev, PARAMS)].sum(), best)


def test_never_contacts_negative_value_customers():
    p = np.array([0.9, 0.01, 0.5])
    rev = np.array([500.0, 20.0, 1.0])
    sel = recommend(p, rev, PARAMS)
    assert sel.tolist() == [True, False, False]


def test_expected_value_beats_raw_risk_when_value_varies():
    p = np.array([0.6, 0.3])
    rev = np.array([30.0, 2000.0])
    sel = recommend(p, rev, PolicyParams(1, 0.3, 40, 12))
    assert sel.tolist() == [False, True]


def test_outcome_accounting():
    sel = np.array([True, True, False, False])
    y = np.array([1, 0, 1, 0])
    rev = np.array([100.0, 100.0, 50.0, 10.0])
    r = outcome(sel, y, rev, PARAMS)
    assert r["churners_reached"] == 1 and r["contacts"] == 2
    assert np.isclose(r["retained_revenue"], 100 * 12 * 0.3)
    assert np.isclose(r["net_value"], 100 * 12 * 0.3 - 80)
    assert np.isclose(r["recall"], 0.5) and np.isclose(r["precision"], 0.5)


def test_heuristic_rules():
    df = pd.DataFrame(
        {
            "days_since_last_login": [3, 40, 3, 40],
            "logins_trend": [1.0, 1.0, 0.2, 0.2],
            "failed_payments_90d": [0, 0, 0, 2],
        }
    )
    assert heuristic_score(df).tolist() == [0, 1, 1, 3]
    assert heuristic_flag(df).tolist() == [0, 1, 1, 1]


def test_flag_top_fraction():
    s = np.array([0.1, 0.9, 0.5, 0.7])
    assert flag_top_fraction(s, 0.5).tolist() == [0, 1, 0, 1]
    assert flag_top_fraction(s, 0.0).sum() == 0
