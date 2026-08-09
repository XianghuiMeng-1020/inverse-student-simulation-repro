"""Metacognitive gap features (dialogue-derived targets + model outputs).

Each scalar is defined as a **proxy** suitable for weak supervision; gold
construction uses two-author blind coding + GPT-4o tie-breaker on a 50-dialog
subset (paper protocol).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class MetacogProfile(BaseModel):
    """Four-dimensional metacognitive gap vector ``g``."""

    monitoring_accuracy: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Proxy for calibration of self-assessment vs eventual correctness. "
            "Operationalization (coding manual): from student turns expressing "
            "certainty/hedging compared against final answer correctness over the "
            "dialogue span; labelers map to [0,1]."
        ),
    )
    help_seeking_ratio: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "#(explicit help/hint requests or 'I don't know') / "
            "#(tutor prompts that afford a student attempt). "
            "Higher means more reliance on external scaffolding."
        ),
    )
    confidence_correctness_gap: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description=(
            "Mean student-stated confidence (mapped to [0,1]) minus mean "
            "correctness indicator on attempts (mapped to [0,1]). Positive "
            "means overconfidence; negative means underconfidence."
        ),
    )
    hint_uptake: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Overlap between tutor hint content and subsequent student turn, "
            "operationalized via token overlap / embedding cosine after stopword "
            "removal; normalized to [0,1]."
        ),
    )


def metacog_feature_names() -> tuple[str, ...]:
    return tuple(MetacogProfile.model_fields.keys())


def zero_metacog_profile() -> MetacogProfile:
    """Neutral profile (used in tests / padding)."""
    return MetacogProfile(
        monitoring_accuracy=0.5,
        help_seeking_ratio=0.0,
        confidence_correctness_gap=0.0,
        hint_uptake=0.0,
    )
