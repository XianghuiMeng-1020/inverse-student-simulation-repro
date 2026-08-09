"""Evaluation metrics (plan p8-01..p8-04)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score


def binary_auc(y_true: Sequence[float | int], y_score: Sequence[float]) -> float:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(y_score, dtype=float)
    if y.size == 0 or len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def binary_brier(y_true: Sequence[float | int], y_prob: Sequence[float]) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_prob, dtype=float)
    if y.size == 0:
        return float("nan")
    return float(brier_score_loss(y, p))


def f1_at_k(y_true: Sequence[int], y_score: Sequence[float], k: int) -> float:
    """Micro-F1 on top-k ranked items (set prediction vs binary gold vector)."""

    y = np.asarray(y_true, dtype=int)
    s = np.asarray(y_score, dtype=float)
    if y.size == 0 or k <= 0:
        return float("nan")
    top = np.argsort(-s)[:k]
    pred = np.zeros_like(y)
    pred[top] = 1
    inter = (pred & y).sum()
    denom = pred.sum() + y.sum()
    if denom == 0:
        return float("nan")
    return float(2 * inter / denom)


def jaccard_at_threshold(y_true: Sequence[int], y_prob: Sequence[float], thr: float = 0.5) -> float:
    y = np.asarray(y_true, dtype=int)
    p = (np.asarray(y_prob, dtype=float) >= thr).astype(int)
    union = np.maximum(y, p).sum()
    if union == 0:
        return float("nan")
    inter = np.minimum(y, p).sum()
    return float(inter / union)


def pearson_r(x: Sequence[float], y: Sequence[float]) -> float:
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    if a.size < 2 or a.size != b.size:
        return float("nan")
    if float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def mae(x: Sequence[float], y: Sequence[float]) -> float:
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    if a.size == 0 or a.size != b.size:
        return float("nan")
    return float(np.mean(np.abs(a - b)))


def ece(probs: Sequence[float], y_true: Sequence[int], *, n_bins: int = 15) -> float:
    """Expected calibration error for binary outcomes."""

    p = np.asarray(probs, dtype=float)
    y = np.asarray(y_true, dtype=int)
    if p.size == 0 or p.size != y.size:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece_sum = 0.0
    for lo, hi in zip(bins[:-1], bins[1:], strict=True):  # noqa: RUF007
        m = (p >= lo) & (p < hi)
        if hi == 1.0:
            m = (p >= lo) & (p <= hi)
        cnt = int(m.sum())
        if cnt == 0:
            continue
        conf = float(p[m].mean())
        acc = float(y[m].mean())
        ece_sum += (cnt / len(p)) * abs(acc - conf)
    return float(ece_sum)


def sharpness(probs: Sequence[float]) -> float:
    """Mean distance of probabilities from 0.5 (Brier sharpness proxy)."""

    p = np.asarray(probs, dtype=float)
    if p.size == 0:
        return float("nan")
    return float(np.mean(np.abs(p - 0.5)))


def mean_bernoulli_entropy(probs: Sequence[float]) -> float:
    """Mean binary entropy for mastery-style probabilities in ``(0,1)``."""

    p = np.clip(np.asarray(probs, dtype=float), 1e-6, 1.0 - 1e-6)
    h = -(p * np.log(p) + (1 - p) * np.log(1 - p))
    return float(np.mean(h))


def mrr_at_k(ranked_indices: Sequence[int], positives: set[int], k: int) -> float:
    """MRR over the first ``k`` ranks (1-indexed reciprocal of first hit)."""

    rr = 0.0
    for rank, idx in enumerate(ranked_indices[:k], start=1):
        if idx in positives:
            rr = 1.0 / rank
            break
    return float(rr)
