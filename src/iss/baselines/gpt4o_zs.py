"""OpenAI GPT-4o zero-shot inverter (plan p7-06)."""

from __future__ import annotations

import json
import os
from typing import Any

from openai import OpenAI

from iss.inverter.prompts import build_inverter_messages
from iss.schema.latent import DialogueTurn


def invert_dialogue_gpt4o(
    turns: list[DialogueTurn],
    *,
    model: str = "gpt-4o",
) -> dict[str, Any]:
    if not os.environ.get("OPENAI_API_KEY"):
        return {"baseline": "gpt4o_zs", "status": "skipped", "reason": "OPENAI_API_KEY unset"}
    client = OpenAI()
    messages = build_inverter_messages(dialogue_turns=turns)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": m["role"], "content": m["content"]} for m in messages],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    raw = (resp.choices[0].message.content or "").strip()
    obj = json.loads(raw)
    return {"baseline": "gpt4o_zs", "status": "ok", "latent": obj, "raw_len": len(raw)}
