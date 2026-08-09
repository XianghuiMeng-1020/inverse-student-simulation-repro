"""Oracle (Z-only) forward SFT rows from MathDial."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from iss.data.mathdial import load_mathdial_hf, row_to_dialogue
from iss.forward.prompts import build_oracle_forward_record
from iss.schema.latent import Dialogue, LatentZ


def _problem_text_from_dialogue(d: Dialogue) -> str:
    if d.metadata and isinstance(d.metadata, dict):
        q = d.metadata.get("question") or d.metadata.get("problem")
        if q:
            return str(q)
    # Fallback: first tutor turn often states the problem
    for t in d.turns:
        if t.speaker == "tutor" and t.text.strip():
            return t.text.strip()[:2000]
    return "Solve the math problem discussed in tutoring."


def iter_mathdial_oracle_forward_records(
    *,
    split: str = "train",
    limit_dialogues: int = 0,
    allowed_dialogue_ids: frozenset[str] | None = None,
    gold_z_by_dialogue_id: dict[str, LatentZ],
) -> Iterator[dict[str, Any]]:
    ds_dict = load_mathdial_hf()

    def emit(d: Dialogue, *, hf_split: str) -> Iterator[dict[str, Any]]:
        z = gold_z_by_dialogue_id.get(d.dialogue_id)
        if z is None:
            return
        problem = _problem_text_from_dialogue(d)
        for t_idx, turn in enumerate(d.turns):
            if turn.speaker != "student":
                continue
            meta = {
                "dialogue_id": d.dialogue_id,
                "split": hf_split,
                "turn_index": str(t_idx),
                "dataset": "mathdial",
                "mode": "oracle_z_only",
            }
            yield build_oracle_forward_record(
                z=z,
                problem_text=problem,
                next_student_text=turn.text,
                meta=meta,
            )

    if allowed_dialogue_ids is None:
        part = ds_dict[split]
        n_dialogues = len(part) if limit_dialogues <= 0 else min(limit_dialogues, len(part))
        for i in range(n_dialogues):
            d = row_to_dialogue(part[i], split=split, idx=i)
            if d is None:
                continue
            yield from emit(d, hf_split=split)
        return

    acc = 0
    for hf_split in ds_dict:
        part = ds_dict[hf_split]
        for i in range(len(part)):
            d = row_to_dialogue(part[i], split=hf_split, idx=i)
            if d is None or d.dialogue_id not in allowed_dialogue_ids:
                continue
            if limit_dialogues > 0:
                if acc >= limit_dialogues:
                    return
                acc += 1
            yield from emit(d, hf_split=hf_split)
