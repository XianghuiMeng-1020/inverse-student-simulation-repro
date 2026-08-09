"""Core latent dialogue schemas for ISS."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from iss.schema.kc_ontology import get_kc_ids
from iss.schema.metacog_features import MetacogProfile
from iss.schema.misconception_catalogue import get_misconception_ids

Speaker = Literal["tutor", "student"]


class DialogueTurn(BaseModel):
    """One utterance in a tutoring transcript."""

    turn_index: int = Field(ge=0)
    speaker: Speaker
    text: str = Field(min_length=1)
    teacher_move: str | None = Field(
        default=None,
        description="Optional MathDial-style teacher move label (if available).",
    )


class Dialogue(BaseModel):
    """Full tutor-student dialogue ``D``."""

    dialogue_id: str = Field(min_length=1)
    turns: list[DialogueTurn] = Field(min_length=1)
    language: str = Field(default="en")
    metadata: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def turns_sorted(self) -> Dialogue:
        idxs = [t.turn_index for t in self.turns]
        if idxs != sorted(idxs):
            msg = "turns must be sorted by turn_index ascending"
            raise ValueError(msg)
        return self


class MasteryVector(BaseModel):
    """``m ∈ [0,1]^{30}`` keyed by canonical KC ids."""

    values: dict[str, float]

    @model_validator(mode="after")
    def validate_keys_and_range(self) -> MasteryVector:
        kcs = set(get_kc_ids())
        keys = set(self.values.keys())
        if keys != kcs:
            missing = sorted(kcs - keys)
            extra = sorted(keys - kcs)
            msg = f"mastery.values must match KC ids exactly; missing={missing}, extra={extra}"
            raise ValueError(msg)
        for kid, v in self.values.items():
            if not 0.0 <= v <= 1.0:
                msg = f"mastery[{kid}]={v} out of [0,1]"
                raise ValueError(msg)
        return self


class MisconceptionState(BaseModel):
    """Multi-hot (soft) misconception vector over |M|=70."""

    probs: dict[str, float]

    @model_validator(mode="after")
    def validate_keys_and_range(self) -> MisconceptionState:
        mids = set(get_misconception_ids())
        keys = set(self.probs.keys())
        if keys != mids:
            missing = sorted(mids - keys)
            extra = sorted(keys - mids)
            msg = f"misconception.probs must match misconception ids exactly; missing={missing}, extra={extra}"
            raise ValueError(msg)
        for mid, p in self.probs.items():
            if not 0.0 <= p <= 1.0:
                msg = f"misconception[{mid}]={p} out of [0,1]"
                raise ValueError(msg)
        return self


class LatentZ(BaseModel):
    """Joint latent ``Z = (m, C, g)`` plus optional rationale string."""

    mastery: MasteryVector
    misconceptions: MisconceptionState
    metacog: MetacogProfile
    rationale: str | None = Field(
        default=None,
        description="Optional short natural-language rationale for debugging / qual analysis.",
    )


class LatentZHard(BaseModel):
    """Discrete misconception set ``C ⊆ M`` (top-k thresholding of probs)."""

    mastery: MasteryVector
    active_misconception_ids: list[str] = Field(default_factory=list)
    metacog: MetacogProfile

    @model_validator(mode="after")
    def validate_active_subset(self) -> LatentZHard:
        mids = set(get_misconception_ids())
        for mid in self.active_misconception_ids:
            if mid not in mids:
                msg = f"unknown misconception id: {mid}"
                raise ValueError(msg)
        return self
