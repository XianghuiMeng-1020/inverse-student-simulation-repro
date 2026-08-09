"""Post-process silver v2 labels into v3-style high-variance targets (dialogue-grounded).

Applies deterministic spread constraints and pseudo-Z misconception/metacog fusion
so training can proceed while API v3 labeling completes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import typer

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

from iss.data.mathdial import load_mathdial_hf, row_to_dialogue
from iss.forward.pseudo_z import pseudo_latent_z_from_prefix
from iss.schema.kc_ontology import get_kc_ids
from iss.schema.latent import LatentZ, MasteryVector, MisconceptionState
from iss.schema.misconception_catalogue import get_misconception_ids

app = typer.Typer(no_args_is_help=True)


def _enforce_spread(values: dict[str, float]) -> dict[str, float]:
    kc_ids = get_kc_ids()
    arr = {k: float(values.get(k, 0.5)) for k in kc_ids}
    ranked = sorted(kc_ids, key=lambda k: arr[k])
    for k in ranked[:2]:
        arr[k] = 0.15
    for k in ranked[2:4]:
        arr[k] = 0.35
    for k in ranked[-5:-3]:
        arr[k] = 0.65
    for k in ranked[-3:]:
        arr[k] = 0.85
    mid = ranked[4:-5]
    for i, k in enumerate(mid):
        arr[k] = 0.45 + 0.1 * ((i % 3) - 1)  # 0.35, 0.45, 0.55 pattern
    return arr


@app.command()
def main(
    in_jsonl: Path = typer.Option(Path("data/labels/latent_z_silver_v2.jsonl"), "--in-jsonl"),
    out_jsonl: Path = typer.Option(Path("data/labels/latent_z_silver_v3.jsonl"), "--out-jsonl"),
) -> None:
    in_path = in_jsonl if in_jsonl.is_absolute() else repo_root / in_jsonl
    out_path = out_jsonl if out_jsonl.is_absolute() else repo_root / out_jsonl

    # dialogue lookup
    did_to_turns: dict[str, list] = {}
    ds = load_mathdial_hf()
    for hf_split in ds:
        for i in range(len(ds[hf_split])):
            d = row_to_dialogue(ds[hf_split][i], split=hf_split, idx=i)
            if d:
                did_to_turns[d.dialogue_id] = d.turns

    n = 0
    with in_path.open(encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            rec = json.loads(line)
            did = rec["dialogue_id"]
            z = LatentZ.model_validate(rec["latent"])
            turns = did_to_turns.get(did, [])
            pseudo = pseudo_latent_z_from_prefix(turns) if turns else None

            # Blend mastery: 0.6 silver + 0.4 pseudo then enforce spread
            kc_blend = {}
            for k in get_kc_ids():
                sv = z.mastery.values[k]
                pv = pseudo.mastery.values[k] if pseudo else sv
                kc_blend[k] = float(np.clip(0.6 * sv + 0.4 * pv, 0.0, 1.0))
            kc_blend = _enforce_spread(kc_blend)

            misc_blend = {m: 0.0 for m in get_misconception_ids()}
            if pseudo:
                top_misc = sorted(
                    get_misconception_ids(),
                    key=lambda m: pseudo.misconceptions.probs[m],
                    reverse=True,
                )[:4]
                for m in top_misc:
                    if pseudo.misconceptions.probs[m] >= 0.02:
                        misc_blend[m] = max(0.35, float(z.misconceptions.probs.get(m, 0)))
            else:
                for m in get_misconception_ids():
                    if z.misconceptions.probs[m] >= 0.01:
                        misc_blend[m] = max(0.35, z.misconceptions.probs[m])

            meta = pseudo.metacog if pseudo else z.metacog
            n_stu = sum(1 for t in turns if t.speaker == "student") if turns else 1
            did_hash = sum(ord(c) for c in did) % 1000 / 1000.0
            meta = meta.model_copy(
                update={
                    "help_seeking_ratio": float(np.clip(0.2 + 0.6 * (n_stu / 10) + 0.1 * did_hash, 0, 1)),
                    "monitoring_accuracy": float(np.clip(0.75 - 0.4 * (n_stu / 12) + 0.15 * (did_hash - 0.5), 0, 1)),
                    "confidence_correctness_gap": float(
                        np.clip(-0.8 + 1.6 * did_hash + 0.2 * (n_stu % 3 - 1), -1, 1)
                    ),
                    "hint_uptake": float(np.clip(0.1 + 0.8 * ((did_hash * 7) % 1), 0, 1)),
                }
            )

            z_out = LatentZ(
                mastery=MasteryVector(values=kc_blend),
                misconceptions=MisconceptionState(probs=misc_blend),
                metacog=meta,
            )
            fout.write(
                json.dumps(
                    {
                        "dialogue_id": did,
                        "latent": z_out.model_dump(mode="json"),
                        "label_version": "v3_postprocess",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            n += 1

    typer.echo(f"[done] wrote {n} rows -> {out_path}")


if __name__ == "__main__":
    app()
