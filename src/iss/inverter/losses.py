"""Multi-term loss placeholders (plan p6-03; wired to training in a later pass)."""

from __future__ import annotations

import torch
import torch.nn.functional as torch_nn_functional


def beta_nll(alpha: torch.Tensor, beta: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Negative log-likelihood under Beta(alpha, beta) for targets in (0,1)."""

    dist = torch.distributions.Beta(alpha.clamp(1e-3, 1e3), beta.clamp(1e-3, 1e3))
    return -dist.log_prob(target.clamp(1e-4, 1 - 1e-4)).mean()


def focal_bce(logits: torch.Tensor, targets: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
    """Multi-label focal BCE (misconception head)."""

    bce = torch_nn_functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    pt = torch.exp(-bce)
    return ((1 - pt) ** gamma * bce).mean()


def huber(pred: torch.Tensor, target: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    return torch_nn_functional.huber_loss(pred, target, reduction="mean", delta=delta)
