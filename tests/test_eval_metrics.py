"""Quick metric sanity checks."""

from __future__ import annotations

from iss.eval.metrics import binary_auc, f1_at_k, mean_bernoulli_entropy


def test_mean_bernoulli_entropy_range() -> None:
    h = mean_bernoulli_entropy([0.5, 0.5, 0.5])
    assert 0.5 < h < 1.0


def test_f1_at_k_perfect() -> None:
    y = [1, 0, 0, 0]
    s = [0.9, 0.1, 0.2, 0.3]
    assert f1_at_k(y, s, k=1) == 1.0


def test_auc_nan_when_single_class() -> None:
    y = [0, 0, 0]
    s = [0.2, 0.3, 0.4]
    assert str(binary_auc(y, s)) == "nan"
