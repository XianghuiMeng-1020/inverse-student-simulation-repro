"""Build inverter SFT rows (full dialogue -> ``LatentZ`` JSON teacher)."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from iss.data.mathdial import load_mathdial_hf, row_to_dialogue
from iss.forward.pseudo_z import pseudo_latent_z_from_prefix
from iss.inverter.prompts import build_inverter_record
from iss.schema.latent import LatentZ


def iter_mathdial_inverter_records(
    *,
    split: str = "train",
    limit_dialogues: int = 0,
    allowed_dialogue_ids: frozenset[str] | None = None,
    gold_z_by_dialogue_id: dict[str, LatentZ] | None = None,
) -> Iterator[dict[str, Any]]:
    """If ``allowed_dialogue_ids`` is set, scan all HF splits (see forward SFT)."""

    ds_dict = load_mathdial_hf()

    if allowed_dialogue_ids is None:
        part = ds_dict[split]
        n = len(part) if limit_dialogues <= 0 else min(limit_dialogues, len(part))
        for i in range(n):
            d = row_to_dialogue(part[i], split=split, idx=i)
            if d is None or len(d.turns) < 2:
                continue
            z = (
                gold_z_by_dialogue_id[d.dialogue_id]
                if gold_z_by_dialogue_id is not None and d.dialogue_id in gold_z_by_dialogue_id
                else pseudo_latent_z_from_prefix(d.turns)
            )
            meta = {"dialogue_id": d.dialogue_id, "split": split, "dataset": "mathdial"}
            yield build_inverter_record(dialogue_turns=d.turns, z=z, meta=meta)
        return

    acc_d = 0
    for hf_split in ds_dict:
        part = ds_dict[hf_split]
        for i in range(len(part)):
            d = row_to_dialogue(part[i], split=hf_split, idx=i)
            if d is None or len(d.turns) < 2:
                continue
            if d.dialogue_id not in allowed_dialogue_ids:
                continue
            if limit_dialogues > 0:
                if acc_d >= limit_dialogues:
                    return
                acc_d += 1
            z = (
                gold_z_by_dialogue_id[d.dialogue_id]
                if gold_z_by_dialogue_id is not None and d.dialogue_id in gold_z_by_dialogue_id
                else pseudo_latent_z_from_prefix(d.turns)
            )
            meta = {"dialogue_id": d.dialogue_id, "split": hf_split, "dataset": "mathdial"}
            yield build_inverter_record(dialogue_turns=d.turns, z=z, meta=meta)
