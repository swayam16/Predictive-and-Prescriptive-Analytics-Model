"""Generate 24 months of operational history for a B2B subscription product and load it into SQLite.

The generator has a hidden state per customer (engagement, a "declining" flag, card risk,
fragility) that drives both what gets recorded (logins, tickets, failed invoices) and the
cancellation hazard. The model only ever sees the recorded tables, so it has to recover the
signal the way it would from a real warehouse: through point-in-time SQL features.

Tables written:
    customers(customer_id, signup_date, plan, contract, channel, region)
    monthly_usage(customer_id, month, logins, active_days, session_minutes,
                  features_used, seats, last_login_at)
    invoices(invoice_id, customer_id, invoice_date, amount, discount_pct, status)
    support_tickets(ticket_id, customer_id, opened_at, priority, resolution_hours, csat)
    cancellations(customer_id, cancelled_at)
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PLANS = np.array(["basic", "pro", "enterprise"])
SEAT_PRICE = {"basic": 15.0, "pro": 35.0, "enterprise": 60.0}
CHANNELS = np.array(["organic", "paid_search", "partner", "outbound"])
REGIONS = np.array(["NA", "EU", "APAC"])

SCHEMA = """
CREATE TABLE customers (
    customer_id  INTEGER PRIMARY KEY,
    signup_date  TEXT NOT NULL,
    plan         TEXT NOT NULL,
    contract     TEXT NOT NULL,
    channel      TEXT NOT NULL,
    region       TEXT NOT NULL
);
CREATE TABLE monthly_usage (
    customer_id     INTEGER NOT NULL REFERENCES customers(customer_id),
    month           TEXT NOT NULL,
    logins          INTEGER NOT NULL,
    active_days     INTEGER NOT NULL,
    session_minutes REAL NOT NULL,
    features_used   INTEGER NOT NULL,
    seats           INTEGER NOT NULL,
    last_login_at   TEXT,
    PRIMARY KEY (customer_id, month)
);
CREATE TABLE invoices (
    invoice_id   INTEGER PRIMARY KEY,
    customer_id  INTEGER NOT NULL REFERENCES customers(customer_id),
    invoice_date TEXT NOT NULL,
    amount       REAL NOT NULL,
    discount_pct REAL NOT NULL,
    status       TEXT NOT NULL CHECK (status IN ('paid', 'failed'))
);
CREATE TABLE support_tickets (
    ticket_id        INTEGER PRIMARY KEY,
    customer_id      INTEGER NOT NULL REFERENCES customers(customer_id),
    opened_at        TEXT NOT NULL,
    priority         TEXT NOT NULL CHECK (priority IN ('low', 'normal', 'high')),
    resolution_hours REAL NOT NULL,
    csat             INTEGER
);
CREATE TABLE cancellations (
    customer_id  INTEGER PRIMARY KEY REFERENCES customers(customer_id),
    cancelled_at TEXT NOT NULL
);
CREATE INDEX ix_usage_month    ON monthly_usage(month);
CREATE INDEX ix_invoices_cust  ON invoices(customer_id, invoice_date);
CREATE INDEX ix_tickets_cust   ON support_tickets(customer_id, opened_at);
"""


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _month_start(start: np.datetime64, m: int | np.ndarray) -> np.ndarray:
    return (start + np.asarray(m).astype("timedelta64[M]")).astype("datetime64[D]")


def simulate(sim: dict[str, Any]) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(sim["seed"])
    n = int(sim["n_customers"])
    n_months = int(sim["n_months"])
    start = np.datetime64(sim["start_month"], "M")
    inc_month = int(sim["price_increase_month"])
    inc_pct = float(sim["price_increase_pct"])

    # ---- static attributes -------------------------------------------------------------
    existing = rng.random(n) < 0.40
    signup_m = np.where(existing, -rng.integers(1, 37, n), rng.integers(0, n_months - 3, n))
    signup_date = _month_start(start, signup_m) + rng.integers(0, 28, n).astype("timedelta64[D]")

    plan = rng.choice(PLANS, n, p=[0.55, 0.35, 0.10])
    seats = np.select(
        [plan == "basic", plan == "pro"],
        [rng.integers(1, 4, n), np.clip(rng.lognormal(1.6, 0.6, n), 2, 30).astype(int)],
        np.clip(rng.lognormal(3.0, 0.6, n), 10, 150).astype(int),
    )
    annual_p = np.select([plan == "basic", plan == "pro"], [0.18, 0.45], 0.80)
    annual = rng.random(n) < annual_p
    channel = rng.choice(CHANNELS, n, p=[0.40, 0.30, 0.15, 0.15])
    region = rng.choice(REGIONS, n, p=[0.50, 0.32, 0.18])
    promo = (channel == "paid_search") & (rng.random(n) < 0.5) & (signup_m >= 0)

    # ---- hidden state ------------------------------------------------------------------
    fragility = rng.normal(0, 1, n)
    card_risk = rng.random(n) < 0.12
    engagement = rng.normal(0.35, 0.75, n) - 0.15 * fragility
    declining = np.zeros(n, dtype=bool)
    unresolved = np.zeros(n, dtype=bool)
    seat_price = np.array([SEAT_PRICE[p] for p in plan])
    # customers who signed up after the list-price change pay the new price from day one
    seat_price = np.where((signup_m >= inc_month) & (plan != "enterprise"), seat_price * (1 + inc_pct), seat_price)
    pending_increase = np.zeros(n, dtype=bool)
    last_fail_m = np.full(n, -99)
    last_jump_m = np.full(n, -99)
    churned = np.zeros(n, dtype=bool)
    cancelled_at = np.full(n, np.datetime64("NaT", "D"), dtype="datetime64[D]")

    usage_rows, invoice_rows, ticket_rows = [], [], []
    ticket_id = invoice_id = 0

    for m in range(n_months):
        alive = (~churned) & (signup_m <= m)
        idx = np.flatnonzero(alive)
        if idx.size == 0:
            continue
        month_start = _month_start(start, m)
        days_in_month = int((_month_start(start, m + 1) - month_start).astype(int))
        tenure = m - signup_m[idx]  # months since signup

        # engagement dynamics
        recent_jump = (m - last_jump_m[idx]) <= 1
        onset_p = _sigmoid(-3.4 + 0.7 * fragility[idx] + 0.7 * unresolved[idx] + 0.6 * recent_jump)
        onset = rng.random(idx.size) < onset_p
        recover = declining[idx] & (rng.random(idx.size) < 0.12)
        declining[idx] = (declining[idx] | onset) & ~recover
        drift = np.where(declining[idx], -0.33, 0.04 * (0.35 - engagement[idx]))
        engagement[idx] += np.where(tenure > 0, drift + rng.normal(0, 0.18, idx.size), 0.0)
        e = engagement[idx]

        # pricing: list increase, renewals, promo expiry
        if m == inc_month:
            eligible = plan[idx] != "enterprise"
            monthly = eligible & ~annual[idx]
            seat_price[idx[monthly]] *= 1 + inc_pct
            last_jump_m[idx[monthly]] = m
            pending_increase[idx[eligible & annual[idx]]] = True
        renewal = annual[idx] & (tenure > 0) & (tenure % 12 == 0) & pending_increase[idx]
        seat_price[idx[renewal]] *= 1 + inc_pct
        last_jump_m[idx[renewal]] = m
        pending_increase[idx[renewal]] = False
        promo_end = promo[idx] & (tenure == 6)
        last_jump_m[idx[promo_end]] = m
        discount = np.where(promo[idx] & (tenure < 6), 0.20, 0.0)

        # seats drift: declining accounts shed seats, healthy ones occasionally add
        shed = declining[idx] & (rng.random(idx.size) < 0.18) & (seats[idx] > 1)
        add = ~declining[idx] & (rng.random(idx.size) < 0.05)
        seats[idx] = seats[idx] - shed * np.maximum(1, (seats[idx] * 0.15).astype(int)) + add

        # invoices
        amount = np.round(seats[idx] * seat_price[idx] * (1 - discount), 2)
        failed = rng.random(idx.size) < (0.012 + 0.11 * card_risk[idx])
        last_fail_m[idx[failed]] = m
        inv_ids = np.arange(invoice_id, invoice_id + idx.size)
        invoice_id += idx.size
        invoice_rows.append(
            pd.DataFrame(
                {
                    "invoice_id": inv_ids,
                    "customer_id": idx,
                    "invoice_date": str(month_start),
                    "amount": amount,
                    "discount_pct": discount,
                    "status": np.where(failed, "failed", "paid"),
                }
            )
        )

        # support tickets
        lam = 0.07 * np.sqrt(seats[idx]) + 0.40 * declining[idx] + 0.30 * ((m - last_jump_m[idx]) <= 1)
        n_tickets = rng.poisson(lam)
        t_cust = np.repeat(idx, n_tickets)
        if t_cust.size:
            t_decl = np.repeat(declining[idx], n_tickets)
            u = rng.random(t_cust.size)
            high_p = 0.18 + 0.20 * t_decl
            priority = np.where(u < high_p, "high", np.where(u < high_p + 0.5, "normal", "low"))
            res_hours = np.round(rng.lognormal(2.9 + 0.5 * t_decl + 0.3 * (priority == "high"), 0.9), 1)
            csat_raw = np.clip(np.round(4.6 - 0.012 * res_hours - 0.8 * t_decl + rng.normal(0, 0.7, t_cust.size)), 1, 5)
            responded = rng.random(t_cust.size) < 0.55
            opened = month_start + rng.integers(0, days_in_month, t_cust.size).astype("timedelta64[D]")
            ticket_rows.append(
                pd.DataFrame(
                    {
                        "ticket_id": np.arange(ticket_id, ticket_id + t_cust.size),
                        "customer_id": t_cust,
                        "opened_at": opened.astype(str),
                        "priority": priority,
                        "resolution_hours": res_hours,
                        "csat": np.where(responded, csat_raw, np.nan),
                    }
                )
            )
            ticket_id += t_cust.size
            slow = pd.Series(res_hours > 96).groupby(t_cust).any()
            unresolved[idx] = False
            unresolved[slow.index[slow.values]] = True
        else:
            unresolved[idx] = False

        # product usage
        logins = rng.poisson(seats[idx] * np.exp(1.3 + 0.75 * e))
        active_days = rng.binomial(21, _sigmoid(0.2 + 1.5 * e))
        logins = np.where(active_days == 0, 0, logins)
        minutes = np.round(logins * rng.gamma(2.0, 6 + 4 * _sigmoid(e)), 1)
        features_used = rng.binomial(12, _sigmoid(-0.3 + 1.1 * e))
        gap = rng.geometric(np.clip(_sigmoid(0.4 + 1.6 * e), 0.02, 0.98)) - 1
        last_day = days_in_month - 1 - gap
        has_login = (logins > 0) & (last_day >= 0)
        last_login = np.where(
            has_login,
            (month_start + np.maximum(last_day, 0).astype("timedelta64[D]")).astype(str),
            None,
        )
        usage_rows.append(
            pd.DataFrame(
                {
                    "customer_id": idx,
                    "month": str(month_start),
                    "logins": np.where(has_login, logins, 0),
                    "active_days": np.where(has_login, active_days, 0),
                    "session_minutes": np.where(has_login, minutes, 0.0),
                    "features_used": np.where(has_login, features_used, 0),
                    "seats": seats[idx],
                    "last_login_at": last_login,
                }
            )
        )

        # cancellation hazard: decided at month end, takes effect during month m + 1
        if m + 1 >= n_months:
            continue
        logit = (
            -3.75
            + 1.05 * np.clip(-e, 0, None)
            + 1.30 * declining[idx]
            + 1.10 * ((m - last_fail_m[idx]) <= 1) * card_risk[idx]
            + 0.65 * unresolved[idx]
            + 0.95 * ((m - last_jump_m[idx]) <= 1)
            + 0.35 * (channel[idx] == "paid_search")
            + 0.25 * (plan[idx] == "basic")
            - 0.035 * np.minimum(tenure, 36)
            + rng.normal(0, 0.6, idx.size)
        )
        at_renewal = ((tenure + 1) % 12) == 0
        logit = np.where(annual[idx], np.where(at_renewal, logit + 0.9, -7.5), logit)
        leaves = rng.random(idx.size) < _sigmoid(logit)
        leavers = idx[leaves]
        churned[leavers] = True
        cancelled_at[leavers] = _month_start(start, m + 1) + rng.integers(0, 28, leavers.size).astype("timedelta64[D]")

    customers = pd.DataFrame(
        {
            "customer_id": np.arange(n),
            "signup_date": signup_date.astype(str),
            "plan": plan,
            "contract": np.where(annual, "annual", "monthly"),
            "channel": channel,
            "region": region,
        }
    )
    cancels = pd.DataFrame({"customer_id": np.flatnonzero(churned), "cancelled_at": cancelled_at[churned].astype(str)})
    tickets = pd.concat(ticket_rows, ignore_index=True) if ticket_rows else pd.DataFrame()
    if not tickets.empty:
        tickets["csat"] = tickets["csat"].astype("Int64")
    return {
        "customers": customers,
        "monthly_usage": pd.concat(usage_rows, ignore_index=True),
        "invoices": pd.concat(invoice_rows, ignore_index=True),
        "support_tickets": tickets,
        "cancellations": cancels,
    }


def write_sqlite(tables: dict[str, pd.DataFrame], db_path: str | Path) -> Path:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    with sqlite3.connect(db_path) as con:
        con.executescript(SCHEMA)
        for name in ["customers", "monthly_usage", "invoices", "support_tickets", "cancellations"]:
            tables[name].to_sql(name, con, if_exists="append", index=False, chunksize=50_000)
    return db_path


def build_warehouse(cfg: dict[str, Any]) -> Path:
    return write_sqlite(simulate(cfg["simulation"]), cfg["paths"]["db"])
