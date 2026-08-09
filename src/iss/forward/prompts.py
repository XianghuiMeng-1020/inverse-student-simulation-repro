"""Prompt templates for conditioning the forward model on ``LatentZ`` + dialogue prefix."""

from __future__ import annotations

import json
from typing import Any

from iss.schema.latent import DialogueTurn, LatentZ

FORWARD_SYSTEM = (
    "You role-play a middle-grades mathematics student in a live tutoring chat. "
    "You must follow the latent state JSON (mastery, misconceptions, metacognition) "
    "when deciding tone, confidence, errors, and help-seeking. "
    "Write ONLY the next student utterance (no role tags, no JSON, no analysis)."
)


def transcript_lines(turns: list[DialogueTurn]) -> str:
    lines: list[str] = []
    for t in turns:
        tag = "Tutor" if t.speaker == "tutor" else "Student"
        lines.append(f"{tag}: {t.text}")
    return "\n".join(lines)


def latent_z_user_block(z: LatentZ) -> str:
    payload = z.model_dump(mode="json", exclude={"rationale"})
    return "Latent state JSON (conditioning; do not repeat verbatim):\n" + json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
    )


def build_forward_messages(
    *,
    z: LatentZ,
    prefix_turns: list[DialogueTurn],
) -> list[dict[str, str]]:
    user = (
        latent_z_user_block(z)
        + "\n\nDialogue so far:\n"
        + transcript_lines(prefix_turns)
        + "\n\nWrite the student's next message only."
    )
    return [
        {"role": "system", "content": FORWARD_SYSTEM},
        {"role": "user", "content": user},
    ]


ORACLE_FORWARD_SYSTEM = (
    "You role-play a middle-grades mathematics student answering a tutor. "
    "You receive ONLY the math problem and a latent-state JSON (mastery, misconceptions, metacognition). "
    "You do NOT see prior dialogue. Generate the student's next plausible utterance "
    "consistent with the latent state (errors, confidence, help-seeking). "
    "Write ONLY the student message (no tags, no JSON)."
)


def build_oracle_forward_messages(
    *,
    z: LatentZ,
    problem_text: str,
) -> list[dict[str, str]]:
    """Z-only forward conditioning (no dialogue prefix) for oracle / bottleneck training."""

    user = (
        latent_z_user_block(z)
        + "\n\nMath problem:\n"
        + problem_text.strip()
        + "\n\nWrite one student utterance that fits this latent state."
    )
    return [
        {"role": "system", "content": ORACLE_FORWARD_SYSTEM},
        {"role": "user", "content": user},
    ]


def build_oracle_forward_record(
    *,
    z: LatentZ,
    problem_text: str,
    next_student_text: str,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "messages": [
            *build_oracle_forward_messages(z=z, problem_text=problem_text),
            {"role": "assistant", "content": next_student_text.strip()},
        ],
    }
    if meta:
        rec["meta"] = meta
    return rec


def build_forward_record(
    *,
    z: LatentZ,
    prefix_turns: list[DialogueTurn],
    next_student_text: str,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "messages": [
            *build_forward_messages(z=z, prefix_turns=prefix_turns),
            {"role": "assistant", "content": next_student_text.strip()},
        ],
    }
    if meta:
        rec["meta"] = meta
    return rec
