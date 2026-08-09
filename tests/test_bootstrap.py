"""Bootstrap CI helpers."""

from __future__ import annotations

import numpy as np

from iss.eval.bootstrap import bootstrap_ci
from iss.eval.metrics import binary_auc, binary_brier


def test_bootstrap_ci_auc_bracket_point() -> None:
    rng = np.random.default_rng(42)
    y = rng.integers(0, 2, size=80)
    p = rng.random(size=80)
    out = bootstrap_ci(y, p, binary_auc, n_boot=400, seed=7)
    assert out["ci_low"] <= out["point"] <= out["ci_high"]


def test_bootstrap_ci_brier_finite() -> None:
    y = np.array([0.0, 1.0, 0.0, 1.0])
    p = np.array([0.1, 0.9, 0.2, 0.8])
    out = bootstrap_ci(y, p, binary_brier, n_boot=200, seed=0)
    assert np.isfinite(out["point"])
    assert np.isfinite(out["ci_low"])
    assert np.isfinite(out["ci_high"])
