"""Permutation importance on the held-out period.

Each feature is shuffled `n_repeats` times and the drop in PR-AUC is recorded. Shuffling
the raw input column (before one-hot encoding) means a categorical such as `contract`
gets a single importance value instead of being split across dummy columns.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from .features import FEATURES


def permutation_table(model: Pipeline, X: pd.DataFrame, y, cfg: dict[str, Any]) -> pd.DataFrame:
    imp = cfg["importance"]
    result = permutation_importance(
        model,
        X[FEATURES],
        y,
        scoring=imp["scoring"],
        n_repeats=imp["n_repeats"],
        random_state=0,
        n_jobs=1,
    )
    table = pd.DataFrame(
        {
            "feature": FEATURES,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False, ignore_index=True)
    total = table["importance_mean"].clip(lower=0).sum()
    table["share"] = table["importance_mean"].clip(lower=0) / total if total > 0 else 0.0
    return table
