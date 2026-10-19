"""Command line: `churnpolicy run` reproduces every number in the README."""

from __future__ import annotations

import argparse
import json

from .config import load_config


def _print_summary(m: dict) -> None:
    h = m["holdout"]["rows"]
    print("\nHeld-out months", ", ".join(m["data"]["test_snapshots"]), f"(base rate {m['holdout']['base_rate']:.1%})")
    print(f"{'':28}{'precision':>10}{'recall':>9}{'F1':>8}{'PR-AUC':>9}{'accuracy':>10}")
    for key, label in [
        ("always_no_churn", "Always 'no churn'"),
        ("heuristic", "Heuristic rules"),
        ("model_matched_volume", "Model, same alert volume"),
        ("model_f1_threshold", "Model, F1 threshold"),
    ]:
        r = h[key]
        print(
            f"{label:28}{r['precision']:>10.3f}{r['recall']:>9.3f}{r['f1']:>8.3f}{r['pr_auc']:>9.3f}"
            f"{r['accuracy']:>10.3f}"
        )
    print(f"\nPolicy at capacity {m['policy']['params']['capacity']} calls/month")
    for r in m["policy"]["summary"]:
        print(
            f"  {r['policy']:16} reached {r['churners_reached']:6.1f} churners  "
            f"net ${r['net_value']:>10,.0f}  ({r['share_of_oracle']:.0%} of ceiling)"
        )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="churnpolicy", description=__doc__)
    p.add_argument("--config", default=None, help="path to config.yaml")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="simulate data, train, evaluate, and write reports")
    r.add_argument("--reuse-data", action="store_true", help="keep the existing SQLite warehouse")
    sub.add_parser("data", help="only (re)build the SQLite warehouse")
    sub.add_parser("summary", help="print the last run's metrics")
    args = p.parse_args(argv)
    cfg = load_config(args.config)

    if args.cmd == "data":
        from .simulate import build_warehouse

        print(build_warehouse(cfg))
    elif args.cmd == "run":
        from .pipeline import run

        _print_summary(run(cfg, rebuild_data=not args.reuse_data))
    elif args.cmd == "summary":
        with open(f"{cfg['paths']['reports']}/metrics.json") as fh:
            _print_summary(json.load(fh))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
