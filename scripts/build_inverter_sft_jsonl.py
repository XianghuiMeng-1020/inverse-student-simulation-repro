"""Write MathDial inverter-SFT JSONL (full dialogue -> ``LatentZ`` JSON)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer

from iss.data.latent_labels import load_latent_z_label_jsonl
from iss.data.splits import load_manifest, manifest_fold_ids
from iss.inverter.sft_rows import iter_mathdial_inverter_records
from iss.schema.latent import LatentZ

app = typer.Typer(no_args_is_help=True)


@app.command()
def main(
    repo_root: Path = typer.Option(Path("."), "--repo-root"),
    split: str = typer.Option("train", "--split"),
    limit_dialogues: int = typer.Option(0, "--limit-dialogues", help="0 = all dialogues in split."),
    out_name: str = typer.Option("mathdial_inverter_train.jsonl", "--out-name"),
    manifest: Path | None = typer.Option(
        None,
        "--manifest",
        help="Optional data/splits/<ds>/manifest.json; filters by --manifest-fold.",
    ),
    manifest_fold: str = typer.Option(
        "train",
        "--manifest-fold",
        help="train | dev | test (only with --manifest).",
    ),
    labels_jsonl: Path | None = typer.Option(
        None,
        "--labels-jsonl",
        help="Optional LLM label file (dialogue_id -> LatentZ); missing ids fall back to pseudo-Z.",
    ),
) -> None:
    allowed = None
    if manifest is not None:
        man_path = manifest if manifest.is_absolute() else (repo_root / manifest).resolve()
        if not man_path.is_file():
            typer.echo(f"[err] missing manifest: {man_path}", err=True)
            raise typer.Exit(code=1)
        allowed = manifest_fold_ids(load_manifest(man_path), manifest_fold)
        typer.echo(f"[info] manifest filter fold={manifest_fold} n_ids={len(allowed)}")

    gold: dict[str, LatentZ] | None = None
    if labels_jsonl is not None:
        lp = labels_jsonl if labels_jsonl.is_absolute() else (repo_root / labels_jsonl).resolve()
        if not lp.is_file():
            typer.echo(f"[err] missing labels: {lp}", err=True)
            raise typer.Exit(code=1)
        gold = load_latent_z_label_jsonl(lp)
        typer.echo(f"[info] loaded labels n={len(gold)} from {lp}")

    out_dir = repo_root / "data" / "inverter_sft"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / out_name
    n = 0
    with out_path.open("w", encoding="utf-8") as f:
        for rec in iter_mathdial_inverter_records(
            split=split,
            limit_dialogues=limit_dialogues,
            allowed_dialogue_ids=allowed,
            gold_z_by_dialogue_id=gold,
        ):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    typer.echo(f"[done] wrote {n} rows -> {out_path}")


if __name__ == "__main__":
    try:
        app()
    except KeyboardInterrupt:
        sys.exit(130)
