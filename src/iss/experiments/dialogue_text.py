"""Turn JSON (from processed parquet) -> plain text features."""

from __future__ import annotations

import json
import re
from typing import Any

_TALKMOVE_SUFFIX = re.compile(r"\s*\[talkmoves_label=[^\]]+\]\s*$")


def loads_turns(turns_json: str) -> list[dict[str, Any]]:
    data = json.loads(turns_json)
    if not isinstance(data, list):
        msg = "turns_json must decode to a list"
        raise ValueError(msg)
    return data


def turns_to_plain_text(
    turns: list[dict[str, Any]], *, strip_talkmoves_suffix: bool = False
) -> str:
    parts: list[str] = []
    for t in turns:
        sp = str(t.get("speaker", "")).strip()
        text = str(t.get("text", "")).strip()
        if strip_talkmoves_suffix:
            text = _TALKMOVE_SUFFIX.sub("", text).strip()
        if not text:
            continue
        parts.append(f"{sp}: {text}")
    return "\n".join(parts)


def extract_last_talkmoves_label(turns: list[dict[str, Any]]) -> str | None:
    pat = re.compile(r"\[talkmoves_label=([^\]]+)\]")
    last: str | None = None
    for t in turns:
        text = str(t.get("text", ""))
        m = pat.search(text)
        if m:
            last = m.group(1).strip()
    return last if last else None


def loads_metadata(metadata_json: str) -> dict[str, Any]:
    data = json.loads(metadata_json)
    if not isinstance(data, dict):
        msg = "metadata_json must decode to an object"
        raise ValueError(msg)
    return data
