"""CIMA (GitHub ``kstats/CIMA`` ``dataset.json``) -> :class:`iss.schema.latent.Dialogue`.

We linearize ``past_convo`` with an even/odd tutor/student heuristic (even
indices tutor, odd student) which matches the paper's alternating pattern in
most exercises. Metadata stores prep/shape subset and exercise id.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from iss.data.parse_common import dialogue_from_turns
from iss.schema.latent import Dialogue, DialogueTurn


def _past_convo_to_turns(past: list[str]) -> list[DialogueTurn]:
    turns: list[DialogueTurn] = []
    for j, line in enumerate(past):
        text = str(line).strip()
        if not text:
            continue
        sp = "tutor" if j % 2 == 0 else "student"
        turns.append(DialogueTurn(turn_index=len(turns), speaker=sp, text=text, teacher_move=None))
    return turns


def iter_cima_dialogues(
    *,
    dataset_path: Path | None = None,
    subsets: tuple[str, ...] = ("prepDataset", "shapeDataset"),
) -> Iterator[Dialogue]:
    root = dataset_path or Path("data/raw/cima/dataset.json")
    data: dict[str, Any] = json.loads(root.read_text(encoding="utf-8"))
    for subset in subsets:
        bucket = data.get(subset) or {}
        for ex_id, payload in bucket.items():
            if not isinstance(payload, dict):
                continue
            past = payload.get("past_convo") or []
            if not isinstance(past, list) or len(past) < 2:
                continue
            turns = _past_convo_to_turns([str(x) for x in past])
            if len(turns) < 2:
                continue
            meta = {
                "dataset": "cima",
                "subset": subset,
                "exercise_id": str(ex_id),
                "language": "it",
            }
            yield dialogue_from_turns(
                f"cima_{subset}_{ex_id}",
                turns,
                language="it",
                metadata=meta,
            )
