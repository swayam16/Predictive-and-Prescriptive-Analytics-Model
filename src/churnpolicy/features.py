"""Turn the warehouse into a modelling table: one row per (snapshot, active customer).

SQL does the point-in-time aggregation (sql/features.sql); pandas adds the ratio and
trend features that are awkward in SQLite and handles missing-value conventions.
"""

from __future__ import annotations

import sqlite3
from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd

CATEGORICAL = ["plan", "contract", "channel", "region"]
NUMERIC = [
    "tenure_months",
    "logins_1m",
    "logins_trend",
    "active_days_1m",
    "active_days_trend",
    "minutes_per_login",
    "features_used_1m",
    "seats_now",
    "seat_change_3m",
    "days_since_last_login",
    "tickets_90d",
    "high_priority_tickets_90d",
    "open_tickets",
    "avg_resolution_hours_90d",
    "avg_csat_90d",
    "has_csat",
    "failed_payments_90d",
    "monthly_revenue",
    "price_change_pct",
    "discount_pct",
    "months_to_renewal",
]
FEATURES = CATEGORICAL + NUMERIC
NO_LOGIN_DAYS = 180.0  # value for "no login in the 6-month lookback"


def _sql(name: str) -> str:
    return resources.files("churnpolicy").joinpath("sql", name).read_text()


def month_range(first: str, last: str) -> list[str]:
    return [d.strftime("%Y-%m-%d") for d in pd.date_range(first, last, freq="MS")]


def raw_features(con: sqlite3.Connection, snapshot: str) -> pd.DataFrame:
    return pd.read_sql_query(_sql("features.sql"), con, params={"snapshot": snapshot})


def derive(raw: pd.DataFrame) -> pd.DataFrame:
    """Ratios, trends and imputation conventions on top of the SQL aggregates."""
    df = raw.copy()
    for col in ["logins_1m", "active_days_1m", "minutes_1m", "features_used_1m"]:
        df[col] = df[col].fillna(0.0)
    df["logins_prev3m_avg"] = df["logins_prev3m_avg"].fillna(df["logins_1m"])
    df["active_days_prev3m_avg"] = df["active_days_prev3m_avg"].fillna(df["active_days_1m"])

    # trend = last month relative to the 3 months before it (1.0 means flat)
    df["logins_trend"] = (df["logins_1m"] + 1) / (df["logins_prev3m_avg"] + 1)
    df["active_days_trend"] = (df["active_days_1m"] + 1) / (df["active_days_prev3m_avg"] + 1)
    df["minutes_per_login"] = np.where(df["logins_1m"] > 0, df["minutes_1m"] / df["logins_1m"].clip(lower=1), 0.0)

    df["seats_now"] = df["seats_now"].fillna(df["seats_3m_ago"]).fillna(1)
    df["seat_change_3m"] = (df["seats_now"] - df["seats_3m_ago"].fillna(df["seats_now"])) / df["seats_now"].clip(
        lower=1
    )

    df["days_since_last_login"] = df["days_since_last_login"].fillna(NO_LOGIN_DAYS).clip(upper=NO_LOGIN_DAYS)

    df["has_csat"] = df["avg_csat_90d"].notna().astype(float)
    df["avg_csat_90d"] = df["avg_csat_90d"].fillna(4.0)  # neutral prior when nobody answered the survey
    df["avg_resolution_hours_90d"] = df["avg_resolution_hours_90d"].fillna(0.0)

    df["monthly_revenue"] = df["monthly_revenue"].fillna(df["amount_3m_ago"]).fillna(0.0)
    prev = df["amount_3m_ago"].fillna(df["monthly_revenue"])
    df["price_change_pct"] = np.where(prev > 0, df["monthly_revenue"] / prev - 1, 0.0)

    tenure_whole = np.floor(df["tenure_months"]).astype(int)
    df["months_to_renewal"] = np.where(df["contract"] == "annual", (12 - tenure_whole % 12) % 12, 0)
    return df


def labels(con: sqlite3.Connection, snapshot: str, label_days: int) -> pd.DataFrame:
    return pd.read_sql_query(_sql("labels.sql"), con, params={"snapshot": snapshot, "label_days": label_days})


def build_snapshot(con: sqlite3.Connection, snapshot: str, label_days: int | None) -> pd.DataFrame:
    df = derive(raw_features(con, snapshot))
    if label_days is not None:
        y = labels(con, snapshot, label_days)
        df = df.merge(y, on="customer_id", how="left", validate="one_to_one")
    keep = ["snapshot", "customer_id", *FEATURES] + (["churned"] if label_days is not None else [])
    return df[keep]


def build_dataset(db_path: str | Path, snapshots: list[str], label_days: int | None) -> pd.DataFrame:
    with sqlite3.connect(db_path) as con:
        frames = [build_snapshot(con, s, label_days) for s in snapshots]
    return pd.concat(frames, ignore_index=True)
