"""Bridge (HF ``rose-e-wang/bridge``) -> :class:`iss.schema.latent.Dialogue`.

Uses the ``c_h`` conversation history list (tutor/student alternating in real
tutoring logs). Weak labels: ``e`` (error type), ``z_what``, ``z_why``.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset

from iss.data.parse_common import dialogue_from_turns, normalize_speaker
from iss.schema.latent import Dialogue, DialogueTurn


def _history_to_turns(c_h: list[dict[str, Any]] | None) -> list[DialogueTurn]:
    if not c_h:
        return []
    turns: list[DialogueTurn] = []
    for msg in c_h:
        uid = str(msg.get("user", "tutor")).strip()
        sp = normalize_speaker(uid)
        text = str(msg.get("text", "")).strip()
        if not text:
            continue
        turns.append(
            DialogueTurn(
                turn_index=len(turns),
                speaker=sp,  # type: ignore[arg-type]
                text=text,
                teacher_move=None,
            )
        )
    return turns


def row_to_dialogue(row: dict[str, Any], *, split: str, idx: int) -> Dialogue | None:
    turns = _history_to_turns(row.get("c_h"))
    if len(turns) < 2:
        return None
    cid = str(row.get("c_id", f"{split}_{idx}"))
    meta = {
        "dataset": "bridge",
        "split": split,
        "error_type": str(row.get("e", "")),
        "z_what": str(row.get("z_what", "")),
        "z_why": str(row.get("z_why", "")),
        "lesson_topic": str(row.get("lesson_topic", "")),
    }
    return dialogue_from_turns(f"bridge_{split}_{cid}", turns, language="en", metadata=meta)


def load_bridge_hf() -> DatasetDict:
    return load_dataset("rose-e-wang/bridge")


def iter_bridge_dialogues(split: str = "train") -> Iterator[Dialogue]:
    ds: DatasetDict = load_bridge_hf()
    part: Dataset = ds[split]
    for i in range(len(part)):
        row = part[i]
        d = row_to_dialogue(row, split=split, idx=i)
        if d is not None:
            yield d
