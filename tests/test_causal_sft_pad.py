"""Pad collator shapes for causal SFT."""

from __future__ import annotations

import torch

from iss.training.causal_sft import pad_batch


def test_pad_batch_shapes() -> None:
    batch = [
        {"input_ids": [1, 2, 3], "labels": [-100, -100, 9]},
        {"input_ids": [4, 5], "labels": [-100, 8]},
    ]
    out = pad_batch(batch, pad_token_id=0, max_length=10)
    assert out["input_ids"].shape == (2, 3)
    assert out["labels"].shape == (2, 3)
    assert (out["labels"][0, :2] == torch.tensor([-100, -100])).all()
    assert out["labels"][0, 2].item() == 9
