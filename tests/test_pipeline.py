"""Smoke test: the whole pipeline runs on a small warehouse and produces sane output."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from churnpolicy.pipeline import run


def test_end_to_end(small_cfg, small_db):
    m = run(small_cfg, rebuild_data=False)
    reports = Path(small_cfg["paths"]["reports"])
    assert json.loads((reports / "metrics.json").read_text())["cv"]["best_model"] in small_cfg["model"]["candidates"]
    for fig in ["cv_metrics", "pr_curve", "permutation_importance", "capacity_curve", "policy_comparison"]:
        assert (reports / "figures" / f"{fig}.png").stat().st_size > 10_000

    rows = m["holdout"]["rows"]
    # the model should beat the heuristic at the same alert volume, even on a small sample
    assert rows["model_matched_volume"]["pr_auc"] > rows["heuristic"]["pr_auc"]
    assert rows["model_matched_volume"]["precision"] > rows["heuristic"]["precision"]

    summary = {r["policy"]: r for r in m["policy"]["summary"]}
    assert summary["expected_value"]["net_value"] >= summary["random"]["net_value"]
    assert summary["oracle"]["net_value"] >= summary["expected_value"]["net_value"]
    for r in summary.values():
        assert r["contacts"] <= small_cfg["policy"]["capacity"]

    calls = pd.read_csv(reports / "call_list.csv")
    assert 0 < len(calls) <= small_cfg["policy"]["capacity"]
    assert calls["expected_value"].is_monotonic_decreasing
