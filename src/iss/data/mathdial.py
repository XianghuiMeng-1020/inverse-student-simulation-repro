"""MathDial (HF ``eth-nlped/mathdial``) -> :class:`iss.schema.latent.Dialogue`."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset

from iss.data.parse_common import dialogue_from_turns, extract_teacher_move, normalize_speaker
from iss.schema.latent import Dialogue, DialogueTurn

EOM = "|EOM|"


def parse_conversation_string(conv: str) -> list[DialogueTurn]:
    """Parse the pipe-delimited MathDial conversation log."""
    parts = [p.strip() for p in conv.split(EOM) if p.strip()]
    turns: list[DialogueTurn] = []
    for part in parts:
        if ":" not in part:
            continue
        head, tail = part.split(":", 1)
        speaker_raw = head.strip()
        text = tail.strip()
        move, body = extract_teacher_move(text)
        sp = normalize_speaker(speaker_raw)
        if sp == "student":
            move = None  # move tags are teacher-only in MathDial
        body = body.strip()
        if not body:
            continue
        turns.append(
            DialogueTurn(
                turn_index=len(turns),
                speaker=sp,  # type: ignore[arg-type]
                text=body,
                teacher_move=move,
            )
        )
    return turns


def row_to_dialogue(row: dict[str, Any], *, split: str, idx: int) -> Dialogue | None:
    turns = parse_conversation_string(row["conversation"])
    if not turns:
        return None
    qid = str(row.get("qid", f"{split}_{idx}"))
    meta = {
        "dataset": "mathdial",
        "split": split,
        "question": str(row.get("question", ""))[:2000],
        "ground_truth": str(row.get("ground_truth", ""))[:2000],
    }
    return dialogue_from_turns(f"mathdial_{split}_{qid}", turns, language="en", metadata=meta)


def load_mathdial_hf() -> DatasetDict:
    return load_dataset("eth-nlped/mathdial")


def iter_mathdial_dialogues(split: str = "train") -> Iterator[Dialogue]:
    ds: DatasetDict = load_mathdial_hf()
    part: Dataset = ds[split]
    for i in range(len(part)):
        row = part[i]
        d = row_to_dialogue(row, split=split, idx=i)
        if d is not None:
            yield d
