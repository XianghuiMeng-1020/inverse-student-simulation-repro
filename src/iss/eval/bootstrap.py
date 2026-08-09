"""Bootstrap confidence intervals for scalar metrics (plan p22-01)."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np


def bootstrap_indices(n: int, rng: np.random.Generator) -> np.ndarray:
    """Row indices with replacement, shape ``(n,)``."""

    return rng.integers(0, n, size=n, endpoint=False)


def bootstrap_ci(
    y: Sequence[float] | np.ndarray,
    p: Sequence[float] | np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    *,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Return point estimate and percentile CI for ``metric_fn(y, p)``.

    ``metric_fn`` must accept 1-D float arrays of equal length. Resamples rows
    independently with replacement (paired bootstrap).
    """

    y_arr = np.asarray(y, dtype=float)
    p_arr = np.asarray(p, dtype=float)
    if y_arr.shape != p_arr.shape or y_arr.ndim != 1:
        msg = "y and p must be 1-D arrays of equal length"
        raise ValueError(msg)
    n = int(y_arr.shape[0])
    if n == 0:
        msg = "empty arrays"
        raise ValueError(msg)
    point = float(metric_fn(y_arr, p_arr))
    rng = np.random.default_rng(seed)
    boots: list[float] = []
    for _ in range(n_boot):
        idx = bootstrap_indices(n, rng)
        boots.append(float(metric_fn(y_arr[idx], p_arr[idx])))
    b = np.asarray(boots, dtype=float)
    lo = float(np.nanquantile(b, alpha / 2))
    hi = float(np.nanquantile(b, 1.0 - alpha / 2))
    return {
        "point": point,
        "ci_low": lo,
        "ci_high": hi,
        "n_boot": float(n_boot),
        "alpha": alpha,
    }
