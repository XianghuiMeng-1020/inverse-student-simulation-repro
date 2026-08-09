"""E7-style probe: zero-shot OpenAI classification vs Bridge gold ``e`` (optional, real API)."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

from openai import OpenAI

from iss.experiments.dialogue_text import loads_metadata, loads_turns, turns_to_plain_text
from iss.experiments.real_benchmarks import _read_shard
from iss.experiments.sklearn_eval import top_k_labels


def run_e7_openai_bridge_error_type(
    processed_root: Path,
    *,
    n_samples: int,
    seed: int,
    model: str,
    e_topk: int,
) -> dict[str, Any]:
    if n_samples <= 0:
        return {
            "task": "e7_openai_bridge_error_type",
            "status": "skipped",
            "reason": "n_samples<=0 (enable with --e7-samples N)",
        }
    if not os.environ.get("OPENAI_API_KEY"):
        return {
            "task": "e7_openai_bridge_error_type",
            "status": "skipped",
            "reason": "OPENAI_API_KEY not set",
        }

    train_df = _read_shard(processed_root / "bridge" / "train.parquet")
    val_df = _read_shard(processed_root / "bridge" / "validation.parquet")
    tr_labels: list[str] = []
    for _, row in train_df.iterrows():
        meta = loads_metadata(str(row["metadata_json"]))
        lab = str(meta.get("error_type", "")).strip()
        if lab:
            tr_labels.append(lab)
    allowed_set = top_k_labels(tr_labels, e_topk)
    allowed = sorted(allowed_set)

    rows: list[tuple[str, str]] = []
    for _, row in val_df.iterrows():
        meta = loads_metadata(str(row["metadata_json"]))
        lab = str(meta.get("error_type", "")).strip()
        if not lab or lab not in allowed_set:
            continue
        turns = loads_turns(str(row["turns_json"]))
        text = turns_to_plain_text(turns, strip_talkmoves_suffix=False)
        if not text.strip():
            continue
        rows.append((text, lab))

    rng = random.Random(seed)
    rng.shuffle(rows)
    rows = rows[:n_samples]

    client = OpenAI()
    correct = 0
    details: list[dict[str, str]] = []
    for text, gold in rows:
        allowed_str = ", ".join(f'"{a}"' for a in allowed)
        prompt = (
            "You are a math tutoring researcher labelling student error types.\n\n"
            f"Allowed labels (exact strings, choose exactly one): [{allowed_str}]\n\n"
            "Tutoring dialogue:\n"
            f"{text[:10000]}\n\n"
            "Instructions:\n"
            "1. Read the dialogue and identify the primary student error.\n"
            "2. Match it to EXACTLY ONE label from the allowed list above.\n"
            "3. Respond with ONLY a JSON object: "
            '{"error_type": "<label from allowed list>"}\n'
            "The value must be copied verbatim from the allowed list."
        )
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You classify student errors in math tutoring. "
                        "Output ONLY a JSON object with key 'error_type'. "
                        "The value MUST be copied verbatim from the allowed list provided."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        raw = (resp.choices[0].message.content or "").strip()
        try:
            obj = json.loads(raw)
            pred = str(obj.get("error_type", "")).strip()
        except json.JSONDecodeError:
            pred = ""
        ok = pred == gold
        correct += int(ok)
        details.append({"gold": gold, "pred": pred, "match": str(ok)})

    acc = correct / len(rows) if rows else 0.0
    return {
        "task": "e7_openai_bridge_error_type",
        "status": "completed",
        "model": model,
        "n_samples": len(rows),
        "e_topk": e_topk,
        "accuracy_vs_gold": acc,
        "allowed_label_count": len(allowed),
        "details": details,
    }


def run_e8_mathtutorbench_placeholder() -> dict[str, Any]:
    return {
        "task": "e8_mathtutorbench_downstream",
        "status": "skipped",
        "reason": "MathTutorBench pipeline not wired in this repo revision",
    }
