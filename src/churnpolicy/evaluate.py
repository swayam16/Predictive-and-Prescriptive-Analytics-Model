"""Cross-validation, holdout evaluation and threshold selection.

Accuracy is reported only to show why it is the wrong headline metric here: with ~6% of
customers churning in a 60-day window, predicting "nobody churns" is ~94% accurate and
useless. Every comparison is therefore made on precision, recall, F1 and PR-AUC, and
model-vs-heuristic comparisons are made at the *same alert volume* so neither side wins
by flagging more customers.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold

from .baseline import heuristic_flag, heuristic_score
from .features import FEATURES
from .models import make_model


def classification_metrics(y: np.ndarray, score: np.ndarray, flag: np.ndarray) -> dict[str, float]:
    y = np.asarray(y)
    flag = np.asarray(flag).astype(int)
    return {
        "precision": float(precision_score(y, flag, zero_division=0)),
        "recall": float(recall_score(y, flag, zero_division=0)),
        "f1": float(f1_score(y, flag, zero_division=0)),
        "accuracy": float(accuracy_score(y, flag)),
        "pr_auc": float(average_precision_score(y, score)),
        "roc_auc": float(roc_auc_score(y, score)) if len(np.unique(y)) > 1 else float("nan"),
        "alert_rate": float(flag.mean()),
    }


def flag_top_fraction(score: np.ndarray, fraction: float) -> np.ndarray:
    """Flag the highest-scoring `fraction` of rows (ties broken by position)."""
    k = int(round(fraction * len(score)))
    flag = np.zeros(len(score), dtype=int)
    if k > 0:
        flag[np.argsort(-score, kind="stable")[:k]] = 1
    return flag


def f1_optimal_threshold(y: np.ndarray, score: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y, score)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    return float(thresholds[int(np.nanargmax(f1[:-1]))])


def cross_validate(train: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, Any]:
    """Customer-grouped stratified k-fold on the training period.

    Grouping by customer keeps every snapshot of a customer in the same fold, so the
    model is never validated on a customer whose other months it trained on.
    """
    X, y, groups = train[FEATURES], train["churned"].to_numpy(), train["customer_id"].to_numpy()
    cv = StratifiedGroupKFold(n_splits=cfg["cv"]["n_splits"], shuffle=True, random_state=cfg["cv"]["seed"])
    names = cfg["model"]["candidates"]
    folds: dict[str, list[dict[str, float]]] = {n: [] for n in ["heuristic", *names]}
    oof = {n: np.zeros(len(train)) for n in names}

    for fold, (tr, va) in enumerate(cv.split(X, y, groups)):
        h_flag = heuristic_flag(X.iloc[va]).to_numpy()
        h_score = heuristic_score(X.iloc[va]).to_numpy()
        folds["heuristic"].append({"fold": fold, **classification_metrics(y[va], h_score, h_flag)})
        alert_rate = h_flag.mean()
        for name in names:
            model = clone(make_model(name, cfg)).fit(X.iloc[tr], y[tr])
            p = model.predict_proba(X.iloc[va])[:, 1]
            oof[name][va] = p
            # same number of alerts as the heuristic raised on this fold
            m = classification_metrics(y[va], p, flag_top_fraction(p, alert_rate))
            folds[name].append({"fold": fold, **m})

    summary = {
        name: {
            k: {"mean": float(np.mean([f[k] for f in rows])), "std": float(np.std([f[k] for f in rows]))}
            for k in rows[0]
            if k != "fold"
        }
        for name, rows in folds.items()
    }
    best = max(names, key=lambda n: summary[n]["pr_auc"]["mean"])
    return {"folds": folds, "summary": summary, "best_model": best, "oof": oof}


def holdout_report(
    y: np.ndarray, proba: np.ndarray, X: pd.DataFrame, threshold: float, snapshots: np.ndarray
) -> dict[str, Any]:
    """Compare the model to the heuristic and to an always-'no churn' baseline on the test period."""
    h_flag = heuristic_flag(X).to_numpy()
    h_score = heuristic_score(X).to_numpy()

    # matched alert volume, computed per decision month so it mirrors how a team would use it
    matched = np.zeros(len(y), dtype=int)
    for s in np.unique(snapshots):
        m = snapshots == s
        matched[m] = flag_top_fraction(proba[m], h_flag[m].mean())

    rows = {
        "always_no_churn": classification_metrics(y, np.zeros(len(y)), np.zeros(len(y))),
        "heuristic": classification_metrics(y, h_score, h_flag),
        "model_matched_volume": classification_metrics(y, proba, matched),
        "model_f1_threshold": classification_metrics(y, proba, (proba >= threshold).astype(int)),
    }
    rows["always_no_churn"]["pr_auc"] = float(y.mean())  # AP of a constant score equals the base rate
    return {
        "base_rate": float(y.mean()),
        "n_rows": int(len(y)),
        "threshold": float(threshold),
        "brier": float(brier_score_loss(y, proba)),
        "rows": rows,
    }
