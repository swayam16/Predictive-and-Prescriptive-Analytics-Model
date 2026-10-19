"""Candidate model pipelines. Each takes the raw feature frame (categoricals as strings)."""

from __future__ import annotations

from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import CATEGORICAL, NUMERIC

LABELS = {
    "heuristic": "Rule-based heuristic",
    "logistic_regression": "Logistic regression",
    "gradient_boosting": "Gradient boosting",
}


def _encoder(scale: bool) -> ColumnTransformer:
    num = StandardScaler() if scale else "passthrough"
    return ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
            ("num", num, NUMERIC),
        ],
        verbose_feature_names_out=False,
    )


def make_model(name: str, cfg: dict[str, Any]) -> Pipeline:
    params = cfg["model"].get(name, {})
    if name == "logistic_regression":
        clf = LogisticRegression(C=params.get("C", 1.0), max_iter=2000)
        return Pipeline([("prep", _encoder(scale=True)), ("clf", clf)])
    if name == "gradient_boosting":
        clf = HistGradientBoostingClassifier(
            learning_rate=params.get("learning_rate", 0.06),
            max_iter=params.get("max_iter", 300),
            max_leaf_nodes=params.get("max_leaf_nodes", 31),
            min_samples_leaf=params.get("min_samples_leaf", 50),
            l2_regularization=params.get("l2_regularization", 0.0),
            early_stopping=False,
            random_state=0,
        )
        return Pipeline([("prep", _encoder(scale=False)), ("clf", clf)])
    raise ValueError(f"unknown model {name!r}")
