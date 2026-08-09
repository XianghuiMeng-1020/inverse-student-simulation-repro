"""TalkMoves (GitHub ``SumnerLab/TalkMoves`` TSV) -> chunked :class:`iss.schema.latent.Dialogue`.

Rows are adjacency-pair annotated utterances without global session ids in the
released TSV. We therefore group **consecutive** ``GROUP`` rows into a single
dialogue for batch training / metacog experiments (documented limitation).
"""

from __future__ import annotations

import csv
from collections.abc import Iterator
from pathlib import Path

from iss.data.parse_common import dialogue_from_turns
from iss.schema.latent import Dialogue, DialogueTurn

DEFAULT_GROUP = 8


def _read_tsv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def iter_talkmoves_dialogues(
    *,
    tsv_path: Path,
    split_name: str,
    role: str,
    group_size: int = DEFAULT_GROUP,
) -> Iterator[Dialogue]:
    rows = _read_tsv_rows(tsv_path)
    sp = "tutor" if role == "teacher" else "student"
    for g in range(0, len(rows), group_size):
        chunk = rows[g : g + group_size]
        turns: list[DialogueTurn] = []
        for r in chunk:
            a = (r.get("text_a") or "").strip()
            b = (r.get("text_b") or "").strip()
            lab = (r.get("labels") or "").strip()
            text = f"{a} // {b}".strip()
            if text == "//":
                continue
            if lab and sp == "student":
                text = f"{text} [talkmoves_label={lab}]"
            turns.append(
                DialogueTurn(
                    turn_index=len(turns),
                    speaker=sp,  # type: ignore[arg-type]
                    text=text,
                    teacher_move=lab if sp == "tutor" else None,
                )
            )
        if len(turns) < 2:
            continue
        did = f"talkmoves_{split_name}_{g // group_size:06d}"
        meta = {
            "dataset": "talkmoves",
            "split": split_name,
            "role_file": role,
            "group_size": str(group_size),
        }
        yield dialogue_from_turns(did, turns, language="en", metadata=meta)


def default_talkmoves_paths(repo_root: Path | None = None) -> dict[str, Path]:
    root = repo_root or Path.cwd()
    base = root / "data" / "raw" / "talkmoves" / "data"
    return {
        "train_teacher": base / "train_teacher.tsv",
        "train_student": base / "train_student.tsv",
        "test_teacher": base / "test_teacher.tsv",
        "test_student": base / "test_student.tsv",
    }
