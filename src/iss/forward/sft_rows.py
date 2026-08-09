"""Build SFT JSONL rows for the forward simulator from MathDial."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from iss.data.mathdial import load_mathdial_hf, row_to_dialogue
from iss.forward.prompts import build_forward_record
from iss.forward.pseudo_z import pseudo_latent_z_from_prefix
from iss.schema.latent import Dialogue, LatentZ


def iter_mathdial_forward_records(
    *,
    split: str = "train",
    limit_dialogues: int = 0,
    limit_steps: int = 0,
    allowed_dialogue_ids: frozenset[str] | None = None,
    gold_z_by_dialogue_id: dict[str, LatentZ] | None = None,
) -> Iterator[dict[str, Any]]:
    """Iterate MathDial rows.

    If ``allowed_dialogue_ids`` is set, every HF split in the dataset dict is
    scanned so ids like ``mathdial_test_<qid>`` in the manifest are reachable.
    """

    ds_dict = load_mathdial_hf()
    steps = 0

    def emit_for_dialogue(d: Dialogue, *, hf_split: str) -> Iterator[dict[str, Any]]:
        nonlocal steps
        gold_z = gold_z_by_dialogue_id.get(d.dialogue_id) if gold_z_by_dialogue_id else None
        for t_idx, turn in enumerate(d.turns):
            if turn.speaker != "student":
                continue
            prefix = d.turns[:t_idx]
            if not prefix:
                continue
            z = gold_z if gold_z is not None else pseudo_latent_z_from_prefix(prefix)
            meta = {
                "dialogue_id": d.dialogue_id,
                "split": hf_split,
                "turn_index": str(t_idx),
                "dataset": "mathdial",
            }
            yield build_forward_record(
                z=z,
                prefix_turns=prefix,
                next_student_text=turn.text,
                meta=meta,
            )
            steps += 1
            if limit_steps > 0 and steps >= limit_steps:
                return

    if allowed_dialogue_ids is None:
        part = ds_dict[split]
        n_dialogues = len(part) if limit_dialogues <= 0 else min(limit_dialogues, len(part))
        for i in range(n_dialogues):
            d = row_to_dialogue(part[i], split=split, idx=i)
            if d is None:
                continue
            yield from emit_for_dialogue(d, hf_split=split)
            if limit_steps > 0 and steps >= limit_steps:
                return
        return

    acc_d = 0
    for hf_split in ds_dict:
        part = ds_dict[hf_split]
        for i in range(len(part)):
            d = row_to_dialogue(part[i], split=hf_split, idx=i)
            if d is None:
                continue
            if d.dialogue_id not in allowed_dialogue_ids:
                continue
            if limit_dialogues > 0:
                if acc_d >= limit_dialogues:
                    return
                acc_d += 1
            yield from emit_for_dialogue(d, hf_split=hf_split)
            if limit_steps > 0 and steps >= limit_steps:
                return
