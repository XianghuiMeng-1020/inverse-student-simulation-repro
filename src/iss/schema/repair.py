"""Coerce common LLM output variations into valid LatentZ dicts (plan p6-02 fallback).

Applied *before* JSON Schema validation so that minor naming deviations from the
model are silently repaired and only structural failures raise errors.
"""

from __future__ import annotations

import re
from typing import Any


def _remap_kc_keys(raw: Any) -> dict[str, float]:
    """Accept mastery.values dicts with keys like KC1 / kc_01 / knowledge_component_1 / KC01."""
    if not isinstance(raw, dict):
        return {}
    out: dict[str, float] = {}
    for k, v in raw.items():
        # Already canonical: KC01..KC30
        if re.fullmatch(r"KC\d{2}", k):
            out[k] = float(v)
            continue
        # KC1..KC9 or KC1..KC30 without leading zero
        m = re.fullmatch(r"[Kk][Cc]_?0*(\d+)", k)
        if m:
            idx = int(m.group(1))
            out[f"KC{idx:02d}"] = float(v)
            continue
        # knowledge_component_N / kc_name_N
        m2 = re.search(r"(\d+)$", k)
        if m2:
            idx = int(m2.group(1))
            out[f"KC{idx:02d}"] = float(v)
    return out


def _remap_misconception_keys(raw: Any) -> dict[str, float]:
    """Accept probs/ids dicts with keys like M1 / M001 / misconception_N / misc_N."""
    if not isinstance(raw, dict):
        return {}
    out: dict[str, float] = {}
    for k, v in raw.items():
        if re.fullmatch(r"M\d{3}", k):
            out[k] = float(v)
            continue
        m = re.fullmatch(r"[Mm]_?0*(\d+)", k)
        if m:
            idx = int(m.group(1))
            out[f"M{idx:03d}"] = float(v)
            continue
        # misconception_N or misc_N
        m2 = re.search(r"(\d+)$", k)
        if m2:
            idx = int(m2.group(1))
            out[f"M{idx:03d}"] = float(v)
    return out


def repair_latent_z_json(data: dict[str, Any]) -> dict[str, Any]:
    """Best-effort normalisation of an LLM-produced LatentZ-like dict.

    Mutates and returns a *new* dict so the original is unchanged.
    """
    from iss.schema.kc_ontology import get_kc_ids
    from iss.schema.misconception_catalogue import get_misconception_ids

    kc_ids = get_kc_ids()
    misc_ids = get_misconception_ids()

    d = dict(data)

    # ── mastery ──────────────────────────────────────────────────────────────
    mastery_raw = d.get("mastery", {})
    if isinstance(mastery_raw, dict):
        # unwrap nested "values" or similar
        values_raw = mastery_raw.get("values", mastery_raw)
        remapped = _remap_kc_keys(values_raw)
        # Fill missing KCs with 0.5
        for kid in kc_ids:
            remapped.setdefault(kid, 0.5)
        # Drop unknown keys
        remapped = {k: remapped[k] for k in kc_ids}
        d["mastery"] = {"values": remapped}

    # ── misconceptions ────────────────────────────────────────────────────────
    misc_raw = d.get("misconceptions", {})
    if isinstance(misc_raw, dict):
        # accept both "probs" and "ids" as the inner dict
        inner = misc_raw.get("probs", misc_raw.get("ids", misc_raw))
        # if the dict itself looks like key->float (no nested key), treat directly
        if all(isinstance(v, (int, float)) for v in inner.values()) if isinstance(inner, dict) else False:
            remapped_m = _remap_misconception_keys(inner)
        else:
            remapped_m = _remap_misconception_keys(inner if isinstance(inner, dict) else misc_raw)
        for mid in misc_ids:
            remapped_m.setdefault(mid, 0.0)
        remapped_m = {k: remapped_m[k] for k in misc_ids}
        d["misconceptions"] = {"probs": remapped_m}

    # ── metacog ───────────────────────────────────────────────────────────────
    metacog_raw = d.get("metacog", {})
    if isinstance(metacog_raw, dict):
        defaults = {
            "monitoring_accuracy": 0.5,
            "help_seeking_ratio": 0.5,
            "confidence_correctness_gap": 0.0,
            "hint_uptake": 0.5,
        }
        mc: dict[str, float] = {}
        for canon, default_v in defaults.items():
            # Try canonical name first
            if canon in metacog_raw:
                mc[canon] = float(metacog_raw[canon])
                continue
            # Try common aliases
            aliases = {
                "monitoring_accuracy": ["monitoring", "accuracy"],
                "help_seeking_ratio": ["help_seeking", "help-seeking"],
                "confidence_correctness_gap": ["confidence", "conf_gap"],
                "hint_uptake": ["hint", "uptake"],
            }
            found = False
            for alias in aliases.get(canon, []):
                for key in metacog_raw:
                    if alias in key.lower():
                        mc[canon] = float(metacog_raw[key])
                        found = True
                        break
                if found:
                    break
            if not found:
                mc[canon] = default_v
        d["metacog"] = mc

    return d
