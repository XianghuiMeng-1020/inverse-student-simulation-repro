"""Tests for silver label JSONL loading."""

from __future__ import annotations

import json
from pathlib import Path

from iss.data.latent_labels import load_latent_z_label_jsonl
from iss.schema.kc_ontology import get_kc_ids
from iss.schema.metacog_features import zero_metacog_profile
from iss.schema.misconception_catalogue import get_misconception_ids


def test_load_latent_z_label_jsonl_roundtrip(tmp_path: Path) -> None:
    kc = {k: 0.5 for k in get_kc_ids()}
    misc = {m: 0.01 for m in get_misconception_ids()}
    z = {
        "mastery": {"values": kc},
        "misconceptions": {"probs": misc},
        "metacog": zero_metacog_profile().model_dump(mode="json"),
    }
    p = tmp_path / "l.jsonl"
    p.write_text(
        json.dumps({"dialogue_id": "mathdial_train_x", "latent": z}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    m = load_latent_z_label_jsonl(p)
    assert len(m) == 1
    assert "mathdial_train_x" in m
    assert m["mathdial_train_x"].mastery.values["KC01"] == 0.5
