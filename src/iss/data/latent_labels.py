"""Load dialogue_id -> :class:`~iss.schema.latent.LatentZ` from JSONL (LLM / expert labels)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from iss.schema.latent import LatentZ


def load_latent_z_label_jsonl(path: Path) -> dict[str, LatentZ]:
    """Parse ``label_latent_z_openai.py`` output: one JSON object per line with ``dialogue_id`` + ``latent``."""

    out: dict[str, LatentZ] = {}
    text = path.read_text(encoding="utf-8")
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        rec: dict[str, Any] = json.loads(line)
        did = str(rec["dialogue_id"])
        out[did] = LatentZ.model_validate(rec["latent"])
    return out
