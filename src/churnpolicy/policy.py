"""Prescriptive layer: turn churn probabilities into a call list under a fixed capacity.

Decision per customer i in a month: contact (1) or not (0), at most `capacity` contacts.

    expected value of contacting i = p_i * revenue_i * horizon * save_rate - contact_cost

Because every contact uses exactly one unit of capacity, maximising total expected value
subject to sum(x_i) <= capacity is solved exactly by sorting on expected value and taking
the top `capacity` customers whose expected value is positive (a unit-weight knapsack).
tests/test_policy.py checks this against brute force.

Ranking by expected value rather than by raw probability matters: a 30% risk on a $2,000
a month account is worth more of a scarce call than a 60% risk on a $30 one.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from .baseline import heuristic_score
from .config import PolicyParams


def expected_value(proba: np.ndarray, revenue: np.ndarray, params: PolicyParams) -> np.ndarray:
    return proba * revenue * params.value_horizon_months * params.save_rate - params.contact_cost


def select_top(priority: np.ndarray, capacity: int, eligible: np.ndarray | None = None) -> np.ndarray:
    """Boolean mask of the `capacity` highest-priority eligible rows."""
    priority = np.asarray(priority, dtype=float)
    mask = np.zeros(len(priority), dtype=bool)
    candidates = np.arange(len(priority)) if eligible is None else np.flatnonzero(eligible)
    if candidates.size == 0 or capacity <= 0:
        return mask
    order = candidates[np.argsort(-priority[candidates], kind="stable")]
    mask[order[:capacity]] = True
    return mask


def recommend(proba: np.ndarray, revenue: np.ndarray, params: PolicyParams) -> np.ndarray:
    ev = expected_value(proba, revenue, params)
    return select_top(ev, params.capacity, eligible=ev > 0)


def outcome(selected: np.ndarray, y: np.ndarray, revenue: np.ndarray, params: PolicyParams) -> dict[str, float]:
    """Realised result of a contact list once the month's true outcomes are known.

    `retained_revenue` is the expected value of saves among contacted customers who
    really were about to churn (save_rate applied to their protected revenue).
    """
    selected = np.asarray(selected, dtype=bool)
    y = np.asarray(y).astype(bool)
    hits = selected & y
    protected = revenue * params.value_horizon_months
    retained = float((protected[hits] * params.save_rate).sum())
    cost = float(selected.sum() * params.contact_cost)
    return {
        "contacts": int(selected.sum()),
        "churners_reached": int(hits.sum()),
        "churners_total": int(y.sum()),
        "precision": float(hits.sum() / max(selected.sum(), 1)),
        "recall": float(hits.sum() / max(y.sum(), 1)),
        "revenue_at_risk_reached": float(protected[hits].sum()),
        "revenue_at_risk_total": float(protected[y].sum()),
        "retained_revenue": retained,
        "cost": cost,
        "net_value": retained - cost,
    }


def _policies(params: PolicyParams, rng: np.random.Generator) -> dict[str, Callable[[pd.DataFrame], np.ndarray]]:
    k = params.capacity

    def random_policy(m: pd.DataFrame) -> np.ndarray:
        return select_top(rng.random(len(m)), k)

    def heuristic_policy(m: pd.DataFrame) -> np.ndarray:
        # the rule flags accounts; among flagged ones, more rules first, then the larger account
        score = heuristic_score(m).to_numpy()
        priority = score * 1e7 + m["monthly_revenue"].to_numpy()
        return select_top(priority, k, eligible=score >= 1)

    def risk_policy(m: pd.DataFrame) -> np.ndarray:
        return select_top(m["proba"].to_numpy(), k)

    def ev_policy(m: pd.DataFrame) -> np.ndarray:
        return recommend(m["proba"].to_numpy(), m["monthly_revenue"].to_numpy(), params)

    def oracle_policy(m: pd.DataFrame) -> np.ndarray:
        # perfect foresight: ceiling for any policy with this capacity
        y = m["churned"].to_numpy().astype(bool)
        ev = np.where(y, m["monthly_revenue"].to_numpy() * params.value_horizon_months * params.save_rate, -1.0)
        ev = ev - params.contact_cost
        return select_top(ev, k, eligible=ev > 0)

    return {
        "random": random_policy,
        "heuristic": heuristic_policy,
        "top_risk": risk_policy,
        "expected_value": ev_policy,
        "oracle": oracle_policy,
    }


POLICY_LABELS = {
    "random": "Random",
    "heuristic": "Heuristic rules",
    "top_risk": "Model: top risk",
    "expected_value": "Model: expected value",
    "oracle": "Perfect foresight",
}


def compare_policies(scored: pd.DataFrame, params: PolicyParams, random_draws: int = 50) -> pd.DataFrame:
    """Average monthly outcome of each policy over the test months.

    `scored` needs columns snapshot, proba, churned, monthly_revenue and the heuristic inputs.
    """
    rows = []
    for snapshot, month in scored.groupby("snapshot", sort=True):
        month = month.reset_index(drop=True)
        y, rev = month["churned"].to_numpy(), month["monthly_revenue"].to_numpy()
        rng = np.random.default_rng(0)
        for name, policy in _policies(params, rng).items():
            draws = random_draws if name == "random" else 1
            results = [outcome(policy(month), y, rev, params) for _ in range(draws)]
            avg = {k: float(np.mean([r[k] for r in results])) for k in results[0]}
            rows.append({"snapshot": snapshot, "policy": name, **avg})
    per_month = pd.DataFrame(rows)
    summary = per_month.drop(columns="snapshot").groupby("policy", sort=False).mean().reset_index()
    oracle_net = float(summary.loc[summary.policy == "oracle", "net_value"].iloc[0])
    summary["share_of_oracle"] = summary["net_value"] / oracle_net if oracle_net else np.nan
    summary["label"] = summary["policy"].map(POLICY_LABELS)
    return summary


def capacity_sweep(scored: pd.DataFrame, params: PolicyParams, capacities: list[int]) -> pd.DataFrame:
    frames = []
    for k in capacities:
        p = PolicyParams(k, params.save_rate, params.contact_cost, params.value_horizon_months)
        frames.append(compare_policies(scored, p, random_draws=20).assign(capacity=k))
    return pd.concat(frames, ignore_index=True)
