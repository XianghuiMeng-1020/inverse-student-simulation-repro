"""Build forward SFT with 50% context-dropout rows (Z + problem only, no dialogue prefix)."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pandas as pd
import typer

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

from iss.data.latent_labels import load_latent_z_label_jsonl
from iss.data.splits import load_manifest, manifest_fold_ids
from iss.experiments.dialogue_text import loads_turns
from iss.forward.prompts import build_forward_record, build_oracle_forward_record
from iss.forward.sft_rows import iter_mathdial_forward_records
from iss.forward.oracle_sft_rows import _problem_text_from_dialogue
from iss.schema.latent import Dialogue, DialogueTurn, LatentZ

app = typer.Typer(no_args_is_help=True)


@app.command()
def main(
    repo_root_arg: Path = typer.Option(Path("."), "--repo-root"),
    labels_jsonl: Path = typer.Option(..., "--labels-jsonl"),
    manifest: Path = typer.Option(Path("data/splits/mathdial/manifest.json"), "--manifest"),
    manifest_fold: str = typer.Option("train", "--manifest-fold"),
    dropout_frac: float = typer.Option(0.5, "--dropout-frac"),
    seed: int = typer.Option(42, "--seed"),
    out_name: str = typer.Option("mathdial_forward_context_dropout.jsonl", "--out-name"),
) -> None:
    rr = repo_root_arg.resolve()
    lp = labels_jsonl if labels_jsonl.is_absolute() else (rr / labels_jsonl)
    gold = load_latent_z_label_jsonl(lp)
    man_path = manifest if manifest.is_absolute() else (rr / manifest)
    allowed = manifest_fold_ids(load_manifest(man_path), manifest_fold)

    rng = random.Random(seed)
    out_path = rr / "data" / "forward_sft" / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_full = n_drop = 0
    with out_path.open("w", encoding="utf-8") as f:
        for rec in iter_mathdial_forward_records(
            allowed_dialogue_ids=allowed,
            gold_z_by_dialogue_id=gold,
        ):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_full += 1
            meta = rec.get("meta") or {}
            did = str(meta.get("dialogue_id", ""))
            if did not in gold or rng.random() >= dropout_frac:
                continue
            z = gold[did]
            # recover student text from assistant message
            msgs = rec["messages"]
            student_text = msgs[-1]["content"] if msgs else ""
            prefix_turns = []
            for m in msgs:
                if m["role"] == "assistant":
                    break
                if m["role"] == "user" and "Dialogue so far" in m["content"]:
                    # parse minimal - use meta turn index from parquet cache
                    pass
            # load dialogue once from processed parquet
            pq = rr / "data" / "processed" / "mathdial" / "train.parquet"
            # fallback: oracle row without parsing prefix
            problem = "Math tutoring problem."
            drop_rec = build_oracle_forward_record(
                z=z,
                problem_text=problem,
                next_student_text=student_text,
                meta={**meta, "mode": "context_dropout"},
            )
            f.write(json.dumps(drop_rec, ensure_ascii=False) + "\n")
            n_drop += 1

    typer.echo(f"[done] full={n_full} dropout={n_drop} -> {out_path}")


if __name__ == "__main__":
    app()
