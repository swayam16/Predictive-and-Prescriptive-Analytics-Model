from __future__ import annotations

import pytest

from churnpolicy.config import load_config
from churnpolicy.simulate import build_warehouse


@pytest.fixture(scope="session")
def small_cfg(tmp_path_factory):
    root = tmp_path_factory.mktemp("run")
    return load_config(
        overrides={
            "paths": {"db": str(root / "w.db"), "artifacts": str(root / "artifacts"), "reports": str(root / "reports")},
            "simulation": {"n_customers": 2500},
            "cv": {"n_splits": 3},
            "model": {"gradient_boosting": {"max_iter": 60}},
            "importance": {"n_repeats": 2},
            "policy": {"capacity": 80, "sweep": [40, 80, 160]},
        }
    )


@pytest.fixture(scope="session")
def small_db(small_cfg):
    return build_warehouse(small_cfg)
