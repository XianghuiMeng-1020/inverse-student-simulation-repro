"""Stratified sample of MathDial test dialogues for expert annotation (n=25)."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pandas as pd
import typer

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

from iss.experiments.dialogue_text import loads_turns

app = typer.Typer(no_args_is_help=True)


@app.command()
def main(
    repo_root_arg: Path = typer.Option(Path("."), "--repo-root"),
    n: int = typer.Option(25, "--n"),
    seed: int = typer.Option(42, "--seed"),
    out_dir: Path = typer.Option(Path("data/labels/expert_pool_test"), "--out-dir"),
) -> None:
    rr = repo_root_arg.resolve()
    df = pd.read_parquet(rr / "data" / "processed" / "mathdial" / "test.parquet")
    rng = random.Random(seed)
    rows = []
    for _, row in df.iterrows():
        turns = loads_turns(str(row["turns_json"]))
        rows.append(
            {
                "dialogue_id": str(row["dialogue_id"]),
                "n_turns": len(turns),
                "n_student": sum(1 for t in turns if t.get("speaker") == "student"),
            }
        )
    rows.sort(key=lambda r: (r["n_turns"], r["dialogue_id"]))
    # stratify by length tertiles
    tert = max(1, len(rows) // 3)
    buckets = [rows[:tert], rows[tert : 2 * tert], rows[2 * tert :]]
    chosen: list[dict] = []
    per_bucket = max(1, n // 3)
    for b in buckets:
        rng.shuffle(b)
        chosen.extend(b[:per_bucket])
    chosen = chosen[:n]

    out = rr / out_dir if not out_dir.is_absolute() else out_dir
    out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for rec in chosen:
        did = rec["dialogue_id"]
        row = df[df["dialogue_id"].astype(str) == did].iloc[0]
        payload = {
            "dialogue_id": did,
            "turns": loads_turns(str(row["turns_json"])),
            "annotation_template": {
                "rater_id": "",
                "mastery": {"values": {}},
                "misconceptions": {"probs": {}},
                "metacog": {},
            },
        }
        (out / f"{did}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        manifest.append(did)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    typer.echo(f"[done] exported {len(manifest)} dialogues -> {out}")


if __name__ == "__main__":
    app()
