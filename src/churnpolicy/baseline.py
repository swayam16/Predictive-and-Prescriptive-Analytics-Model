"""The business-rule heuristic the model has to beat.

This mirrors the kind of rule a customer-success team typically runs from a dashboard:
flag an account if it has gone quiet, its usage has halved, or a payment has failed.
The score is the number of rules triggered (0-3), which also gives the heuristic a
ranking so it can be compared under the same contact capacity as the model.
"""

from __future__ import annotations

import pandas as pd

INACTIVE_DAYS = 21
USAGE_DROP = 0.5


def rule_hits(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "inactive": df["days_since_last_login"] > INACTIVE_DAYS,
            "usage_drop": df["logins_trend"] < USAGE_DROP,
            "payment_failed": df["failed_payments_90d"] >= 1,
        },
        index=df.index,
    )


def heuristic_score(df: pd.DataFrame) -> pd.Series:
    """Number of rules triggered, 0 to 3."""
    return rule_hits(df).sum(axis=1).astype(float)


def heuristic_flag(df: pd.DataFrame) -> pd.Series:
    """1 if any rule fires."""
    return (heuristic_score(df) >= 1).astype(int)
