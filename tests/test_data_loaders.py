"""Unit tests for dialogue parsers (no large raw files required)."""

from __future__ import annotations

from pathlib import Path

import pytest

from iss.data.bridge import _history_to_turns
from iss.data.mathdial import parse_conversation_string
from iss.schema.latent import Dialogue


def test_mathdial_parse_moves() -> None:
    s = (
        "Teacher: (probing)If you had 4 and tripled, how much?|EOM|"
        "Steven: 12.|EOM|Teacher: (generic)Good."
    )
    turns = parse_conversation_string(s)
    assert [t.speaker for t in turns] == ["tutor", "student", "tutor"]
    assert turns[0].teacher_move == "probing"
    assert "tripled" in turns[0].text


def test_bridge_history_turns() -> None:
    c_h = [
        {"user": "tutor", "text": "What is the area?"},
        {"user": "student", "text": "4 m"},
    ]
    turns = _history_to_turns(c_h)
    assert len(turns) == 2
    assert turns[0].speaker == "tutor"


@pytest.mark.skipif(
    not Path("data/raw/cima/dataset.json").exists(),
    reason="CIMA raw JSON not present (run scripts/download_data.py)",
)
def test_cima_yields_dialogue() -> None:
    from iss.data.cima import iter_cima_dialogues

    d = next(iter_cima_dialogues())
    assert isinstance(d, Dialogue)
    assert d.language == "it"


@pytest.mark.skipif(
    not Path("data/raw/talkmoves/data/train_student.tsv").exists(),
    reason="TalkMoves raw TSV not present",
)
def test_talkmoves_iter() -> None:
    from iss.data.talkmoves import iter_talkmoves_dialogues

    p = Path("data/raw/talkmoves/data/train_student.tsv")
    d = next(iter_talkmoves_dialogues(tsv_path=p, split_name="train_student", role="student"))
    assert len(d.turns) >= 2
