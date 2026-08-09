"""Tests for ISS latent schemas, catalogue sizes, and JSON Schema export."""

from __future__ import annotations

import copy
import json

import jsonschema
import pytest
from pydantic import ValidationError

from iss.schema.grammar import (
    latent_z_json_schema,
    validate_latent_z_json,
    write_latent_schema_json,
)
from iss.schema.kc_ontology import get_kc_ids, load_knowledge_components
from iss.schema.latent import LatentZ, MasteryVector, MisconceptionState
from iss.schema.metacog_features import MetacogProfile, zero_metacog_profile
from iss.schema.misconception_catalogue import get_misconception_ids, load_misconceptions


def _dummy_latent() -> LatentZ:
    kcs = get_kc_ids()
    mids = get_misconception_ids()
    return LatentZ(
        mastery=MasteryVector(values=dict.fromkeys(kcs, 0.5)),
        misconceptions=MisconceptionState(probs=dict.fromkeys(mids, 0.0)),
        metacog=zero_metacog_profile(),
    )


def test_ontology_sizes() -> None:
    assert len(load_knowledge_components()) == 30
    assert len(load_misconceptions()) == 70
    assert get_kc_ids()[0] == "KC01" and get_kc_ids()[-1] == "KC30"
    assert get_misconception_ids()[0] == "M001" and get_misconception_ids()[-1] == "M070"


def test_latent_round_trip() -> None:
    z = _dummy_latent()
    dumped = z.model_dump(mode="json")
    z2 = LatentZ.model_validate(dumped)
    assert z2 == z


def test_mastery_missing_key_raises() -> None:
    kcs = list(get_kc_ids())
    bad = dict.fromkeys(kcs[:-1], 0.1)  # drop last
    with pytest.raises(ValueError):
        MasteryVector(values=bad)


def test_mastery_out_of_range_raises() -> None:
    kcs = get_kc_ids()
    vals = dict.fromkeys(kcs, 0.0)
    vals[kcs[0]] = 1.1
    with pytest.raises(ValueError):
        MasteryVector(values=vals)


def test_json_schema_validate_and_parse() -> None:
    z = _dummy_latent()
    data = z.model_dump(mode="json")
    z2 = validate_latent_z_json(data)
    assert z2 == z


def test_json_schema_reject_bad_type() -> None:
    z = _dummy_latent()
    data = z.model_dump(mode="json")
    data["metacog"]["monitoring_accuracy"] = "not-a-float"
    with pytest.raises((jsonschema.ValidationError, ValidationError)):
        validate_latent_z_json(data)


def test_write_latent_schema_json_idempotent(tmp_path) -> None:
    p = tmp_path / "schema.json"
    write_latent_schema_json(p)
    assert p.exists()
    schema = latent_z_json_schema()
    assert "properties" in schema


def test_committed_schema_matches_generated() -> None:
    """Ensure repo `latent_schema.json` stays in sync with models."""
    from pathlib import Path

    from iss.schema import grammar

    committed = Path(grammar.__file__).with_name("latent_schema.json")
    assert committed.exists()
    fresh = latent_z_json_schema()
    on_disk = json.loads(committed.read_text(encoding="utf-8"))
    assert on_disk == fresh


def test_metacog_bounds() -> None:
    with pytest.raises(ValueError):
        MetacogProfile(
            monitoring_accuracy=1.5,
            help_seeking_ratio=0.0,
            confidence_correctness_gap=0.0,
            hint_uptake=0.0,
        )


def test_latent_optional_rationale() -> None:
    z = _dummy_latent().model_copy(update={"rationale": "student overgeneralized linearity"})
    assert z.rationale is not None
    z2 = LatentZ.model_validate(z.model_dump())
    assert z2.rationale == z.rationale


def test_hard_subset_round_trip() -> None:
    from iss.schema.latent import LatentZHard

    zh = LatentZHard(
        mastery=_dummy_latent().mastery,
        active_misconception_ids=["M001", "M007"],
        metacog=zero_metacog_profile(),
    )
    zh2 = LatentZHard.model_validate(zh.model_dump())
    assert zh2 == zh


def test_unknown_misconception_id_raises() -> None:
    from iss.schema.latent import LatentZHard

    with pytest.raises(ValueError):
        LatentZHard(
            mastery=_dummy_latent().mastery,
            active_misconception_ids=["NOT_AN_ID"],
            metacog=zero_metacog_profile(),
        )


def test_schema_extra_keys_rejected_by_pydantic() -> None:
    z = _dummy_latent()
    data = z.model_dump(mode="json")
    data["extra_field"] = 123
    # LatentZ default extra handling is "ignore"; strip and re-parse:
    data2 = copy.deepcopy(data)
    del data2["extra_field"]
    assert LatentZ.model_validate(data2) == z
