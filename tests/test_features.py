"""The features must be computable on the snapshot date. These tests try to break that."""

from __future__ import annotations

import shutil
import sqlite3

import numpy as np
import pandas as pd

from churnpolicy.features import FEATURES, build_snapshot

SNAPSHOT = "2025-09-01"


def _features(db, snapshot=SNAPSHOT, label_days=None):
    with sqlite3.connect(db) as con:
        return build_snapshot(con, snapshot, label_days).set_index("customer_id")


def test_deleting_the_future_changes_nothing(small_db, tmp_path):
    before = _features(small_db)
    pruned = tmp_path / "pruned.db"
    shutil.copy(small_db, pruned)
    with sqlite3.connect(pruned) as con:
        con.execute("DELETE FROM monthly_usage WHERE month >= ?", (SNAPSHOT,))
        con.execute("DELETE FROM invoices WHERE invoice_date >= ?", (SNAPSHOT,))
        con.execute("DELETE FROM support_tickets WHERE opened_at >= ?", (SNAPSHOT,))
        con.execute("DELETE FROM cancellations WHERE cancelled_at >= ?", (SNAPSHOT,))
    after = _features(pruned)
    pd.testing.assert_frame_equal(before[FEATURES], after.loc[before.index, FEATURES])


def test_ticket_outcomes_after_the_snapshot_are_not_used(small_db, tmp_path):
    before = _features(small_db)
    changed = tmp_path / "changed.db"
    shutil.copy(small_db, changed)
    with sqlite3.connect(changed) as con:
        # tickets still open on the snapshot: rewrite how they eventually turned out
        con.execute(
            """UPDATE support_tickets SET resolution_hours = resolution_hours + 500, csat = 1
               WHERE opened_at < ? AND julianday(opened_at) + resolution_hours / 24.0 >= julianday(?)""",
            (SNAPSHOT, SNAPSHOT),
        )
    pd.testing.assert_frame_equal(before[FEATURES], _features(changed)[FEATURES])


def test_population_is_active_customers_with_history(small_db):
    df = _features(small_db)
    with sqlite3.connect(small_db) as con:
        cust = pd.read_sql("SELECT * FROM customers", con).set_index("customer_id")
        canc = pd.read_sql("SELECT * FROM cancellations", con).set_index("customer_id")
    assert (cust.loc[df.index, "signup_date"] < "2025-08-01").all()
    cancelled_before = canc.index[canc.cancelled_at < SNAPSHOT]
    assert not set(cancelled_before) & set(df.index)


def test_label_is_cancellation_inside_window(small_db):
    df = _features(small_db, label_days=60)
    with sqlite3.connect(small_db) as con:
        canc = pd.read_sql("SELECT * FROM cancellations", con).set_index("customer_id")["cancelled_at"]
    when = canc.reindex(df.index)
    expected = ((when >= SNAPSHOT) & (when < "2025-10-31")).astype(int)
    np.testing.assert_array_equal(df["churned"].to_numpy(), expected.to_numpy())
    assert 0.01 < df["churned"].mean() < 0.2


def test_features_have_no_missing_values(small_db):
    df = _features(small_db)
    assert not df[FEATURES].isna().any().any()
