"""End-to-end run: warehouse -> features -> CV -> holdout -> importance -> policy -> artifacts."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score

from . import plots
from .baseline import heuristic_score
from .config import PolicyParams
from .evaluate import cross_validate, f1_optimal_threshold, holdout_report
from .features import FEATURES, build_dataset, month_range
from .importance import permutation_table
from .models import make_model
from .policy import capacity_sweep, compare_policies, expected_value, recommend
from .simulate import build_warehouse


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_datasets(cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    s = cfg["snapshots"]
    db = cfg["paths"]["db"]
    labelled = build_dataset(db, month_range(s["first"], s["last_labelled"]), s["label_days"])
    train = labelled[labelled.snapshot <= s["train_last"]].reset_index(drop=True)
    test = labelled[labelled.snapshot >= s["test_first"]].reset_index(drop=True)
    live = build_dataset(db, [s["score"]], label_days=None)
    return labelled, train, test, live


def run(cfg: dict[str, Any], rebuild_data: bool = True) -> dict[str, Any]:
    paths = {k: Path(v) for k, v in cfg["paths"].items()}
    figs = paths["reports"] / "figures"
    for p in (paths["artifacts"], figs):
        p.mkdir(parents=True, exist_ok=True)

    if rebuild_data or not paths["db"].exists():
        _log("simulating 24 months of history into SQLite")
        build_warehouse(cfg)

    _log("extracting point-in-time features with SQL")
    labelled, train, test, live = load_datasets(cfg)
    _log(f"train {len(train):,} rows / test {len(test):,} rows / live {len(live):,} customers")

    _log(f"{cfg['cv']['n_splits']}-fold customer-grouped cross-validation")
    cv = cross_validate(train, cfg)
    best = cv["best_model"]
    threshold = f1_optimal_threshold(train["churned"].to_numpy(), cv["oof"][best])
    _log(f"best by CV PR-AUC: {best}; F1-optimal threshold from out-of-fold predictions: {threshold:.3f}")

    _log("fitting on the training period and scoring the held-out months")
    fitted = {n: make_model(n, cfg).fit(train[FEATURES], train["churned"]) for n in cfg["model"]["candidates"]}
    test_proba = {n: m.predict_proba(test[FEATURES])[:, 1] for n, m in fitted.items()}
    model = fitted[best]
    y_test = test["churned"].to_numpy()
    holdout = holdout_report(y_test, test_proba[best], test[FEATURES], threshold, test["snapshot"].to_numpy())
    holdout["by_model_pr_auc"] = {
        n: holdout_report(y_test, p, test[FEATURES], threshold, test["snapshot"].to_numpy())["rows"][
            "model_matched_volume"
        ]["pr_auc"]
        for n, p in test_proba.items()
    }

    _log("permutation importance on the held-out months")
    imp = permutation_table(model, test, y_test, cfg)

    _log("evaluating contact policies under the capacity constraint")
    params = PolicyParams.from_config(cfg)
    scored = test.assign(proba=test_proba[best])
    policies = compare_policies(scored, params)
    sweep = capacity_sweep(scored, params, cfg["policy"]["sweep"])

    _log("refitting on every labelled month and building the live call list")
    final = make_model(best, cfg).fit(labelled[FEATURES], labelled["churned"])
    live = live.assign(proba=final.predict_proba(live[FEATURES])[:, 1])
    live["expected_value"] = expected_value(live["proba"].to_numpy(), live["monthly_revenue"].to_numpy(), params)
    live["recommended"] = recommend(live["proba"].to_numpy(), live["monthly_revenue"].to_numpy(), params)
    live["heuristic_rules"] = heuristic_score(live).astype(int)
    calls = (
        live[live.recommended]
        .sort_values("expected_value", ascending=False)
        .assign(rank=lambda d: np.arange(1, len(d) + 1))
    )
    call_cols = [
        "rank",
        "customer_id",
        "plan",
        "contract",
        "monthly_revenue",
        "proba",
        "expected_value",
        "days_since_last_login",
        "logins_trend",
        "failed_payments_90d",
        "open_tickets",
        "heuristic_rules",
    ]

    _log("writing figures, metrics and artifacts")
    heuristic_pts = []
    h = heuristic_score(test).to_numpy()
    for k in (1, 2):
        flag = (h >= k).astype(int)
        heuristic_pts.append((recall_score(y_test, flag), precision_score(y_test, flag, zero_division=0), f"≥{k}"))
    plots.cv_metrics(cv["summary"], figs / "cv_metrics.png")
    plots.pr_curves(
        y_test,
        {n: test_proba[n] for n in ["gradient_boosting", "logistic_regression"] if n in test_proba},
        heuristic_pts,
        figs / "pr_curve.png",
    )
    plots.calibration(y_test, test_proba[best], figs / "calibration.png")
    plots.importance(imp, figs / "permutation_importance.png")
    plots.capacity_curve(sweep, params.capacity, figs / "capacity_curve.png")
    plots.policy_bars(policies, figs / "policy_comparison.png")

    joblib.dump(
        {"model": final, "features": FEATURES, "trained_through": cfg["snapshots"]["last_labelled"]},
        paths["artifacts"] / "model.joblib",
    )
    scored[["snapshot", "customer_id", "proba", "churned", "monthly_revenue", *FEATURES]].to_csv(
        paths["artifacts"] / "holdout_scores.csv.gz", index=False
    )
    live.to_csv(paths["artifacts"] / "live_scores.csv.gz", index=False)
    calls[call_cols].to_csv(paths["reports"] / "call_list.csv", index=False, float_format="%.4f")
    imp.to_csv(paths["reports"] / "permutation_importance.csv", index=False, float_format="%.5f")
    sweep.to_csv(paths["reports"] / "capacity_sweep.csv", index=False, float_format="%.2f")

    metrics = {
        "data": {
            "customers": int(labelled.customer_id.nunique()),
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "train_snapshots": [train.snapshot.min(), train.snapshot.max()],
            "test_snapshots": sorted(test.snapshot.unique().tolist()),
            "train_base_rate": float(train.churned.mean()),
            "live_snapshot": cfg["snapshots"]["score"],
            "live_customers": int(len(live)),
        },
        "cv": {"best_model": best, "summary": cv["summary"], "folds": cv["folds"]},
        "holdout": holdout,
        "importance": imp.head(10).to_dict(orient="records"),
        "policy": {
            "params": params.__dict__,
            "summary": policies.drop(columns="label").to_dict(orient="records"),
        },
        "live": {
            "recommended": int(live.recommended.sum()),
            "expected_value": float(live.loc[live.recommended, "expected_value"].sum()),
        },
    }
    with open(paths["reports"] / "metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2, default=float)
    _log("done")
    return metrics
