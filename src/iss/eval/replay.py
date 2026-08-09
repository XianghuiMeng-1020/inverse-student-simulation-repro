"""Counterfactual forward-replay diagnostics (plan p10-02)."""

from __future__ import annotations

import math
from typing import Any

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from iss.forward.prompts import build_forward_messages
from iss.schema.latent import DialogueTurn, LatentZ
from iss.training.causal_sft import ChatSFTDataset


@torch.inference_mode()
def teacher_forced_nll(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    *,
    messages: list[dict[str, str]],
    max_length: int,
) -> float:
    """Mean NLL (natural log) over supervised assistant tokens only."""

    ds = ChatSFTDataset([{"messages": messages}], tokenizer, max_length=max_length)
    row = ds[0]
    dev = next(model.parameters()).device
    batch = {
        "input_ids": torch.tensor([row["input_ids"]], device=dev),
        "attention_mask": torch.tensor([([1] * len(row["input_ids"]))], device=dev),
        "labels": torch.tensor([row["labels"]], device=dev),
    }
    out = model(**batch)
    return float(out.loss)


def counterfactual_replay_row(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    *,
    z: LatentZ,
    prefix_turns: list[DialogueTurn],
    gold_next_student: str,
    max_length: int,
) -> dict[str, Any]:
    """Gold continuation NLL under ``(Z, \\text{prefix})`` conditioning."""

    msgs = [
        *build_forward_messages(z=z, prefix_turns=prefix_turns),
        {"role": "assistant", "content": gold_next_student.strip()},
    ]
    nll = teacher_forced_nll(model, tokenizer, messages=msgs, max_length=max_length)
    ppl = math.exp(nll) if nll < 20 else float("inf")
    return {
        "nll_mean": nll,
        "ppl": ppl,
        "n_prefix_turns": len(prefix_turns),
    }


def student_turn_indices(turns: list[DialogueTurn]) -> list[int]:
    """Indices of student utterances."""

    return [i for i, t in enumerate(turns) if t.speaker == "student"]


def mean_nll_next_k_student_turns(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    *,
    z: LatentZ,
    turns: list[DialogueTurn],
    start_student_index: int,
    horizon: int,
    max_length: int,
) -> float | None:
    """Mean teacher-forced NLL over the next ``horizon`` *student* turns (gold prefixes).

    ``start_student_index`` must be the dialogue index of a student turn. Uses the
    same ``z`` for each step (prefix grows with gold transcript). Returns ``None`` if
    fewer than ``horizon`` future student turns exist from that anchor.
    """

    stud = student_turn_indices(turns)
    if start_student_index not in stud:
        return None
    j = stud.index(start_student_index)
    chunk = stud[j : j + horizon]
    if len(chunk) < horizon:
        return None
    nlls: list[float] = []
    for ti in chunk:
        prefix = turns[:ti]
        gold = turns[ti].text
        row = counterfactual_replay_row(
            model,
            tokenizer,
            z=z,
            prefix_turns=prefix,
            gold_next_student=gold,
            max_length=max_length,
        )
        nlls.append(float(row["nll_mean"]))
    return sum(nlls) / len(nlls)
