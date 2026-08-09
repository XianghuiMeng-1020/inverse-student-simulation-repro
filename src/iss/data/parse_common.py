"""Shared parsing helpers."""

from __future__ import annotations

import re
from typing import Any

from iss.schema.latent import Dialogue, DialogueTurn


def normalize_speaker(raw: str) -> str:
    r = raw.strip().lower()
    if r in {"teacher", "tutor", "t"}:
        return "tutor"
    if r in {"student", "s", "learner"}:
        return "student"
    # MathDial uses student first names (e.g. "Steven"); Bridge uses tutor/student tokens.
    return "student"


_MOVE_PAREN = re.compile(r"^\(([^)]*)\)\s*(.*)$", re.DOTALL)


def extract_teacher_move(text: str) -> tuple[str | None, str]:
    """If text begins with ``(move)rest``, return (move, rest)."""
    m = _MOVE_PAREN.match(text.strip())
    if not m:
        return None, text.strip()
    return m.group(1).strip() or None, m.group(2).strip()


def dialogue_from_turns(
    dialogue_id: str,
    turns: list[DialogueTurn],
    *,
    language: str = "en",
    metadata: dict[str, str] | None = None,
) -> Dialogue:
    return Dialogue(
        dialogue_id=dialogue_id,
        turns=turns,
        language=language,
        metadata=metadata or {},
    )


def turns_to_jsonable(turns: list[DialogueTurn]) -> list[dict[str, Any]]:
    return [t.model_dump(mode="json") for t in turns]
