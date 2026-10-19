"""Interactive demo: tune the retention policy and see the call list and its back-tested value.

Run locally with `streamlit run app/streamlit_app.py`. Reads the artifacts that
`churnpolicy run` writes; if they are missing it runs the pipeline once first.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from churnpolicy.config import PolicyParams, load_config  # noqa: E402
from churnpolicy.policy import POLICY_LABELS, compare_policies, expected_value, recommend  # noqa: E402

st.set_page_config(page_title="Retention Policy Model", page_icon="📉", layout="wide")
CFG = load_config(ROOT / "config.yaml")
ART = ROOT / CFG["paths"]["artifacts"]
REP = ROOT / CFG["paths"]["reports"]


@st.cache_resource(show_spinner="First run: simulating data and training the model (about 2 minutes)…")
def ensure_artifacts() -> bool:
    if not (ART / "live_scores.csv.gz").exists():
        import os

        from churnpolicy.pipeline import run

        os.chdir(ROOT)
        run(CFG)
    return True


@st.cache_data
def load() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    live = pd.read_csv(ART / "live_scores.csv.gz")
    holdout = pd.read_csv(ART / "holdout_scores.csv.gz")
    metrics = json.loads((REP / "metrics.json").read_text())
    return live, holdout, metrics


@st.cache_data
def backtest(capacity: int, save_rate: float, cost: float, horizon: int) -> pd.DataFrame:
    _, holdout, _ = load()
    return compare_policies(holdout, PolicyParams(capacity, save_rate, cost, horizon), random_draws=10)


ensure_artifacts()
live, holdout, metrics = load()

st.title("Predictive & prescriptive retention model")
st.caption(
    f"Gradient-boosted churn model trained on {metrics['data']['train_snapshots'][0]} to "
    f"{metrics['data']['train_snapshots'][1]} snapshots, validated on "
    f"{', '.join(s[:7] for s in metrics['data']['test_snapshots'])}, and scoring "
    f"{metrics['data']['live_customers']:,} active customers as of {metrics['data']['live_snapshot']}. "
    "All data is synthetic."
)

with st.sidebar:
    st.header("Policy settings")
    capacity = st.slider("Retention calls per month", 50, 2000, CFG["policy"]["capacity"], step=50)
    save_rate = st.slider(
        "Save rate when a would-be churner is called", 0.05, 0.8, float(CFG["policy"]["save_rate"]), step=0.05
    )
    cost = st.number_input("Cost per call ($)", 0, 1000, int(CFG["policy"]["contact_cost"]), step=10)
    horizon = st.slider("Months of revenue a save protects", 1, 36, CFG["policy"]["value_horizon_months"])
params = PolicyParams(capacity, save_rate, float(cost), horizon)

tab_calls, tab_backtest, tab_model = st.tabs(["Call list", "Back-test", "Model validation"])

with tab_calls:
    ev = expected_value(live["proba"].to_numpy(), live["monthly_revenue"].to_numpy(), params)
    picked = recommend(live["proba"].to_numpy(), live["monthly_revenue"].to_numpy(), params)
    calls = live.assign(expected_value=ev)[picked].sort_values("expected_value", ascending=False)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers to call", f"{len(calls):,}")
    c2.metric("Expected net value", f"${calls.expected_value.sum():,.0f}")
    c3.metric("Average churn risk on list", f"{calls.proba.mean():.0%}" if len(calls) else "–")
    c4.metric("Not flagged by the old rules", f"{(calls.heuristic_rules == 0).mean():.0%}" if len(calls) else "–")
    if len(calls) < capacity:
        st.info(
            f"Only {len(calls)} customers have positive expected value at these settings, "
            "so the policy leaves the remaining capacity unused."
        )
    show = calls[
        [
            "customer_id",
            "plan",
            "contract",
            "monthly_revenue",
            "proba",
            "expected_value",
            "days_since_last_login",
            "logins_trend",
            "failed_payments_90d",
            "heuristic_rules",
        ]
    ]
    st.dataframe(
        show,
        hide_index=True,
        use_container_width=True,
        column_config={
            "customer_id": "Customer",
            "monthly_revenue": st.column_config.NumberColumn("Monthly revenue", format="$%.0f"),
            "proba": st.column_config.ProgressColumn("Churn risk (60 days)", min_value=0, max_value=1, format="%.2f"),
            "expected_value": st.column_config.NumberColumn("Expected value of call", format="$%.0f"),
            "days_since_last_login": st.column_config.NumberColumn("Days since login", format="%.0f"),
            "logins_trend": st.column_config.NumberColumn("Login trend", format="%.2f"),
            "failed_payments_90d": "Failed payments (90d)",
            "heuristic_rules": "Old rules fired",
        },
    )
    st.download_button("Download call list (CSV)", show.to_csv(index=False), "call_list.csv", "text/csv")

with tab_backtest:
    st.write(
        "Each policy picks who to call in each held-out month using only what was known on the 1st; "
        "the result is then scored against who actually cancelled."
    )
    bt = backtest(capacity, save_rate, float(cost), horizon)
    order = ["random", "heuristic", "top_risk", "expected_value", "oracle"]
    bt = bt.set_index("policy").loc[order].reset_index()
    view = pd.DataFrame(
        {
            "Policy": bt.policy.map(POLICY_LABELS),
            "Churners reached / month": bt.churners_reached.round(1),
            "Precision of calls": bt.precision.map("{:.1%}".format),
            "Revenue at risk reached": bt.revenue_at_risk_reached.map("${:,.0f}".format),
            "Net value / month": bt.net_value.map("${:,.0f}".format),
            "Share of ceiling": bt.share_of_oracle.map("{:.0%}".format),
        }
    )
    st.dataframe(view, hide_index=True, use_container_width=True)
    st.bar_chart(
        pd.DataFrame({"Policy": view["Policy"], "Net value ($/month)": bt.net_value}).set_index("Policy"),
        horizontal=True,
        color="#2a78d6",
    )
    st.image(
        str(REP / "figures" / "capacity_curve.png"), caption="Capacity sweep at the default settings from config.yaml"
    )

with tab_model:
    rows = metrics["holdout"]["rows"]
    table = pd.DataFrame(
        [
            {
                "Approach": label,
                "Precision": rows[k]["precision"],
                "Recall": rows[k]["recall"],
                "F1": rows[k]["f1"],
                "PR-AUC": rows[k]["pr_auc"],
                "Accuracy": rows[k]["accuracy"],
                "Alert rate": rows[k]["alert_rate"],
            }
            for k, label in [
                ("always_no_churn", "Always predict 'no churn'"),
                ("heuristic", "Heuristic rules"),
                ("model_matched_volume", "Model, same alert volume as rules"),
                ("model_f1_threshold", "Model, F1-optimal threshold"),
            ]
        ]
    )
    st.subheader("Held-out months")
    st.dataframe(
        table.style.format({c: "{:.3f}" for c in table.columns if c != "Approach"}),
        hide_index=True,
        use_container_width=True,
    )
    st.caption("Accuracy is shown only to make the point: predicting nobody churns scores highest on it.")
    a, b = st.columns(2)
    a.image(str(REP / "figures" / "cv_metrics.png"))
    b.image(str(REP / "figures" / "pr_curve.png"))
    a.image(str(REP / "figures" / "permutation_importance.png"))
    b.image(str(REP / "figures" / "calibration.png"))
