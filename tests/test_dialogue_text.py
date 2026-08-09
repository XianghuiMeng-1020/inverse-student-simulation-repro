"""Unit tests for dialogue text helpers (no embedding / API)."""

from __future__ import annotations

import json

from iss.experiments.dialogue_text import (
    extract_last_talkmoves_label,
    loads_turns,
    turns_to_plain_text,
)


def test_strip_talkmoves_for_features() -> None:
    turns = [
        {"speaker": "student", "text": "I think x=2 [talkmoves_label=Reasoning]"},
        {"speaker": "student", "text": "Wait no [talkmoves_label=Uncertainty]"},
    ]
    plain = turns_to_plain_text(turns, strip_talkmoves_suffix=True)
    assert "talkmoves_label" not in plain
    assert "Reasoning" not in plain
    assert "Wait no" in plain


def test_extract_last_talkmoves_label() -> None:
    turns = json.loads(
        json.dumps(
            [
                {"speaker": "student", "text": "a [talkmoves_label=Foo]"},
                {"speaker": "student", "text": "b [talkmoves_label=Bar]"},
            ],
        ),
    )
    assert extract_last_talkmoves_label(turns) == "Bar"


def test_loads_turns_roundtrip() -> None:
    raw = [{"speaker": "tutor", "text": "Hi"}]
    t = loads_turns(json.dumps(raw))
    assert t[0]["text"] == "Hi"
