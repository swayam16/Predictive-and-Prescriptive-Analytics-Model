"""Matplotlib figures for the README and the report.

One shared style: light surface, recessive grid, fixed categorical colour order, a legend
whenever there is more than one series, and text in ink colours rather than series colours.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402

from .models import LABELS  # noqa: E402
from .policy import POLICY_LABELS  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#e87ba4"]
NEUTRAL = "#9a9893"

# colour follows the entity everywhere
MODEL_COLORS = {"heuristic": NEUTRAL, "logistic_regression": SERIES[1], "gradient_boosting": SERIES[0]}
POLICY_COLORS = {
    "expected_value": SERIES[0],
    "top_risk": SERIES[1],
    "heuristic": SERIES[2],
    "random": NEUTRAL,
    "oracle": INK_2,
}


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 12.5,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.titlepad": 12,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "legend.labelcolor": INK_2,
            "font.family": "DejaVu Sans",
            "lines.linewidth": 2,
        }
    )


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def cv_metrics(summary: dict, out: Path) -> Path:
    _style()
    metrics = [("precision", "Precision"), ("recall", "Recall"), ("f1", "F1"), ("pr_auc", "PR-AUC")]
    names = ["heuristic", *[n for n in summary if n != "heuristic"]]
    x = np.arange(len(metrics))
    width = 0.8 / len(names)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for i, name in enumerate(names):
        means = [summary[name][m]["mean"] for m, _ in metrics]
        stds = [summary[name][m]["std"] for m, _ in metrics]
        pos = x - 0.4 + width * (i + 0.5)
        ax.bar(pos, means, width * 0.92, color=MODEL_COLORS[name], label=LABELS[name], zorder=3)
        ax.errorbar(pos, means, yerr=stds, fmt="none", ecolor=INK_2, elinewidth=1, capsize=2.5, zorder=4)
        for px, v in zip(pos, means, strict=True):
            ax.text(px, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=7.5, color=INK_2)
    ax.set_xticks(x, [label for _, label in metrics])
    ax.set_ylim(0, 1)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Score (mean of 5 folds, ±1 sd)")
    ax.set_title("Cross-validation at the same alert volume as the heuristic")
    ax.legend(loc="upper left", ncols=3)
    return _save(fig, out)


def pr_curves(
    y: np.ndarray, probas: dict[str, np.ndarray], heuristic_points: list[tuple[float, float, str]], out: Path
) -> Path:
    _style()
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    for name, p in probas.items():
        precision, recall, _ = precision_recall_curve(y, p)
        ax.plot(recall, precision, color=MODEL_COLORS[name], label=LABELS[name], zorder=3)
    for i, (r, p, label) in enumerate(heuristic_points):
        ax.scatter(
            [r],
            [p],
            s=60,
            color=NEUTRAL,
            edgecolor=SURFACE,
            linewidth=2,
            zorder=5,
            label="Heuristic (≥1, ≥2 rules)" if i == 0 else None,
        )
        ax.annotate(label, (r, p), xytext=(6, 6), textcoords="offset points", fontsize=8, color=INK_2)
    ax.axhline(y.mean(), color=INK_2, linewidth=1, linestyle=(0, (4, 3)), zorder=2)
    ax.text(
        0.99,
        y.mean() + 0.012,
        f"base rate {y.mean():.1%}",
        ha="right",
        fontsize=8,
        color=INK_2,
        transform=ax.get_yaxis_transform(),
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–recall on the held-out months")
    ax.legend(loc="upper right")
    return _save(fig, out)


def calibration(y: np.ndarray, proba: np.ndarray, out: Path) -> Path:
    _style()
    frac_pos, mean_pred = calibration_curve(y, proba, n_bins=12, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5, 4.6))
    lim = max(mean_pred.max(), frac_pos.max()) * 1.08
    ax.plot([0, lim], [0, lim], color=INK_2, linewidth=1, linestyle=(0, (4, 3)), label="Perfect calibration")
    ax.plot(
        mean_pred,
        frac_pos,
        color=SERIES[0],
        marker="o",
        markersize=5,
        markeredgecolor=SURFACE,
        markeredgewidth=1.5,
        label="Gradient boosting",
    )
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Predicted churn probability")
    ax.set_ylabel("Observed churn rate")
    ax.set_title("Calibration on held-out months")
    ax.legend(loc="upper left")
    return _save(fig, out)


def importance(table: pd.DataFrame, out: Path, top: int = 14) -> Path:
    _style()
    t = table.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.2, 0.36 * len(t) + 1.2))
    ax.barh(
        t["feature"],
        t["importance_mean"],
        xerr=t["importance_std"],
        color=SERIES[0],
        height=0.62,
        error_kw={"ecolor": INK_2, "elinewidth": 1, "capsize": 2},
        zorder=3,
    )
    for yi, (v, sd) in enumerate(zip(t["importance_mean"], t["importance_std"], strict=True)):
        ax.text(v + sd + t["importance_mean"].max() * 0.015, yi, f"{v:.3f}", va="center", fontsize=8, color=INK_2)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Drop in PR-AUC when the feature is shuffled (mean of 10 repeats)")
    ax.set_title("Permutation importance, held-out months")
    ax.set_xlim(0, (t["importance_mean"] + t["importance_std"]).max() * 1.15)
    return _save(fig, out)


def capacity_curve(sweep: pd.DataFrame, chosen: int, out: Path) -> Path:
    _style()
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    order = ["oracle", "expected_value", "top_risk", "heuristic", "random"]
    for name in order:
        s = sweep[sweep.policy == name].sort_values("capacity")
        style = {"linestyle": (0, (4, 3)), "linewidth": 1.5} if name == "oracle" else {}
        ax.plot(
            s["capacity"],
            s["net_value"] / 1e3,
            color=POLICY_COLORS[name],
            label=POLICY_LABELS[name],
            marker="o" if name != "oracle" else None,
            markersize=4,
            **style,
        )
    ax.axvline(chosen, color=INK_2, linewidth=1, linestyle=(0, (2, 3)))
    ax.text(
        chosen, 1.0, f"  capacity = {chosen}", transform=ax.get_xaxis_transform(), va="top", fontsize=8, color=INK_2
    )
    ax.axhline(0, color=INK_2, linewidth=0.8)
    ax.set_xlabel("Retention calls available per month")
    ax.set_ylabel("Net value per month ($k)")
    ax.set_title("Net retained revenue by policy and capacity")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncols=5, fontsize=8.5)
    return _save(fig, out)


def policy_bars(summary: pd.DataFrame, out: Path) -> Path:
    _style()
    s = summary.set_index("policy").loc[["random", "heuristic", "top_risk", "expected_value", "oracle"]].iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    colors = [POLICY_COLORS[p] for p in s.index]
    vals = s["net_value"] / 1e3
    ax.barh([POLICY_LABELS[p] for p in s.index], vals, color=colors, height=0.6, zorder=3)
    span = vals.max() - min(vals.min(), 0)
    for yi, (v, share) in enumerate(zip(vals, s["share_of_oracle"], strict=True)):
        ax.text(
            max(v, 0) + span * 0.015,
            yi,
            f"{'−' if v < 0 else ''}${abs(v):,.0f}k  ({share:.0%} of ceiling)",
            va="center",
            fontsize=8.5,
            color=INK_2,
        )
    ax.axvline(0, color=INK_2, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(min(vals.min(), 0) * 1.1, vals.max() * 1.45)
    ax.set_xlabel("Net value per month ($k), averaged over test months")
    ax.set_title("Policy comparison at the fixed capacity")
    return _save(fig, out)
