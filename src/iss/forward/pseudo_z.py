"""Weak ``Z*`` targets for early-phase SFT (replace with LLM/human gold later)."""

from __future__ import annotations

import re

import numpy as np

from iss.schema.kc_ontology import get_kc_ids
from iss.schema.latent import DialogueTurn, LatentZ, MasteryVector, MisconceptionState
from iss.schema.metacog_features import MetacogProfile, zero_metacog_profile
from iss.schema.misconception_catalogue import get_misconception_ids

_FRACTION_KCS = frozenset({"KC07", "KC08", "KC09", "KC10"})
_HELP_TOKENS = (
    "don't know",
    "dont know",
    "idk",
    "help",
    "hint",
    "confused",
    "stuck",
    "not sure",
)


def _concat_student_text(turns: list[DialogueTurn]) -> str:
    return " ".join(t.text.lower() for t in turns if t.speaker == "student")


def _metacog_from_prefix(turns: list[DialogueTurn]) -> MetacogProfile:
    stu = [t for t in turns if t.speaker == "student"]
    if not stu:
        return zero_metacog_profile()
    blob = _concat_student_text(turns)
    asks = sum(1 for k in _HELP_TOKENS if k in blob)
    help_seeking_ratio = min(1.0, asks / max(1, len(stu)))
    # crude proxy: hedging without correctness supervision -> keep mid-band
    monitoring_accuracy = max(0.0, min(1.0, 0.65 - 0.15 * help_seeking_ratio))
    confidence_correctness_gap = min(1.0, max(-1.0, 0.25 * help_seeking_ratio))
    hint_uptake = 0.35 if ("hint" in blob or "because" in blob) else 0.15
    return MetacogProfile(
        monitoring_accuracy=monitoring_accuracy,
        help_seeking_ratio=help_seeking_ratio,
        confidence_correctness_gap=confidence_correctness_gap,
        hint_uptake=min(1.0, max(0.0, hint_uptake)),
    )


def pseudo_latent_z_from_prefix(prefix_turns: list[DialogueTurn]) -> LatentZ:
    """Heuristic ``Z`` from dialogue prefix (weak supervision bootstrap)."""

    blob = " ".join(t.text.lower() for t in prefix_turns)
    kc_vals = dict.fromkeys(get_kc_ids(), 0.5)
    if re.search(r"fraction|numerator|denominator|/\d|\d/", blob):
        for kid in _FRACTION_KCS:
            kc_vals[kid] = 0.62
    if any(w in blob for w in ("equation", "variable", "x =", "solve for")):
        kc_vals["KC17"] = min(0.85, kc_vals.get("KC17", 0.5) + 0.12)
    mids = dict.fromkeys(get_misconception_ids(), 0.02)
    if "same as" in blob and any(ch.isdigit() for ch in blob):
        mids["M001"] = 0.18
    return LatentZ(
        mastery=MasteryVector(values=kc_vals),
        misconceptions=MisconceptionState(probs=mids),
        metacog=_metacog_from_prefix(prefix_turns),
        rationale=None,
    )


def uniform_random_latent_z(*, seed: int) -> LatentZ:
    """IID Uniform[0,1] mastery / misc / metacog (E4 random-``Z'`` control arm)."""

    rng = np.random.default_rng(seed & 0xFFFFFFFF)
    kc_vals = {kid: float(rng.uniform(0.0, 1.0)) for kid in get_kc_ids()}
    mids = {mid: float(rng.uniform(0.0, 1.0)) for mid in get_misconception_ids()}
    meta = MetacogProfile(
        monitoring_accuracy=float(rng.uniform(0.0, 1.0)),
        help_seeking_ratio=float(rng.uniform(0.0, 1.0)),
        confidence_correctness_gap=float(rng.uniform(-1.0, 1.0)),
        hint_uptake=float(rng.uniform(0.0, 1.0)),
    )
    return LatentZ(
        mastery=MasteryVector(values=kc_vals),
        misconceptions=MisconceptionState(probs=mids),
        metacog=meta,
        rationale=None,
    )
