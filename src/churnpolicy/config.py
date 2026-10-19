"""Load config.yaml into a plain dict with a couple of typed helpers."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "config.yaml"


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    with open(path or DEFAULT_CONFIG) as fh:
        cfg = yaml.safe_load(fh)
    if overrides:
        cfg = deep_update(cfg, overrides)
    return cfg


def deep_update(base: dict, updates: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


@dataclass(frozen=True)
class PolicyParams:
    capacity: int
    save_rate: float
    contact_cost: float
    value_horizon_months: int

    @classmethod
    def from_config(cls, cfg: dict[str, Any], **overrides: Any) -> PolicyParams:
        p = cfg["policy"]
        values = {
            "capacity": p["capacity"],
            "save_rate": p["save_rate"],
            "contact_cost": p["contact_cost"],
            "value_horizon_months": p["value_horizon_months"],
        }
        values.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**values)
