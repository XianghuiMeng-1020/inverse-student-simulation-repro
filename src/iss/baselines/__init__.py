"""Baseline registry (plan p7-08)."""

from __future__ import annotations

import json
from typing import Any

from iss.schema.grammar import validate_latent_z_json
from iss.schema.latent import LatentZ


def parse_latent_z_json(text: str) -> LatentZ:
    s = text.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[-1]
        if "```" in s:
            s = s.rsplit("```", 1)[0]
    data = json.loads(s)
    return validate_latent_z_json(data)


def run_baseline(name: str, **kwargs: Any) -> dict[str, Any]:
    if name == "gpt4o_zs":
        from iss.baselines.gpt4o_zs import invert_dialogue_gpt4o

        return invert_dialogue_gpt4o(**kwargs)
    msg = f"unknown baseline '{name}'"
    raise KeyError(msg)


__all__ = ["parse_latent_z_json", "run_baseline"]
