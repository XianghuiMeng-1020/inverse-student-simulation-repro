"""Inverter prompts: dialogue -> structured ``LatentZ`` JSON."""

from __future__ import annotations

import json

from iss.schema.latent import DialogueTurn, LatentZ

INVERTER_SYSTEM = (
    "You estimate a student's latent cognitive state from a tutoring dialogue.\n"
    "Reply with ONE JSON object matching exactly this structure (no markdown fences):\n"
    "{\n"
    '  "mastery": {"values": {"KC01": 0.5, "KC02": 0.5, ..., "KC30": 0.5}},\n'
    '  "misconceptions": {"probs": {"M001": 0.0, "M002": 0.0, ..., "M070": 0.0}},\n'
    '  "metacog": {"monitoring_accuracy": 0.5, "help_seeking_ratio": 0.5,\n'
    '              "confidence_correctness_gap": 0.0, "hint_uptake": 0.5}\n'
    "}\n"
    "Keys must be exactly KC01..KC30 and M001..M070. All values in [0,1]."
)


def transcript_lines(turns: list[DialogueTurn]) -> str:
    lines: list[str] = []
    for t in turns:
        tag = "Tutor" if t.speaker == "tutor" else "Student"
        lines.append(f"{tag}: {t.text}")
    return "\n".join(lines)


def build_inverter_messages(*, dialogue_turns: list[DialogueTurn]) -> list[dict[str, str]]:
    user = "Full dialogue:\n" + transcript_lines(dialogue_turns) + "\n\nEmit LatentZ JSON only."
    return [
        {"role": "system", "content": INVERTER_SYSTEM},
        {"role": "user", "content": user},
    ]


def build_inverter_record(
    *,
    dialogue_turns: list[DialogueTurn],
    z: LatentZ,
    meta: dict[str, str] | None = None,
) -> dict[str, object]:
    assistant = json.dumps(
        z.model_dump(mode="json", exclude={"rationale"}),
        ensure_ascii=False,
    )
    rec: dict[str, object] = {
        "messages": [
            *build_inverter_messages(dialogue_turns=dialogue_turns),
            {"role": "assistant", "content": assistant},
        ],
    }
    if meta:
        rec["meta"] = meta
    return rec
