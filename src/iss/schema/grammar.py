"""JSON Schema export + validation helpers for structured LLM outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema
from pydantic import TypeAdapter

from iss.schema.latent import LatentZ

_SCHEMA_PATH = Path(__file__).with_name("latent_schema.json")


def latent_z_json_schema() -> dict[str, Any]:
    """Return the JSON Schema for :class:`iss.schema.latent.LatentZ`."""
    return TypeAdapter(LatentZ).json_schema()


def write_latent_schema_json(path: Path | None = None) -> Path:
    """Write ``latent_schema.json`` next to this module (or to ``path``)."""
    target = path or _SCHEMA_PATH
    target.write_text(json.dumps(latent_z_json_schema(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return target


def validate_latent_z_json(data: dict[str, Any]) -> LatentZ:
    """Validate ``data`` against JSON Schema then parse into :class:`LatentZ`."""
    schema = latent_z_json_schema()
    jsonschema.validate(instance=data, schema=schema)
    return LatentZ.model_validate(data)


def load_cached_schema_dict() -> dict[str, Any]:
    """Load committed schema JSON (fast path for inference servers)."""
    if not _SCHEMA_PATH.exists():
        write_latent_schema_json()
    return json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
