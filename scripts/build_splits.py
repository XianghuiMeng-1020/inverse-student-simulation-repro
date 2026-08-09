"""Write per-corpus train/dev/test manifests under ``data/splits/``."""

from __future__ import annotations

import sys
from pathlib import Path

import typer

from iss.data.registry import CORE_DATASETS
from iss.data.splits import build_manifest, write_manifest

app = typer.Typer(no_args_is_help=True)


@app.command()
def main(
    repo_root: Path = typer.Option(Path("."), "--repo-root"),
    seed: int = typer.Option(42, "--seed"),
    datasets: str = typer.Option(
        ",".join(CORE_DATASETS),
        "--datasets",
        help=f"Comma subset of {CORE_DATASETS}",
    ),
) -> None:
    processed = repo_root / "data" / "processed"
    splits_root = repo_root / "data" / "splits"
    names = [x.strip() for x in datasets.split(",") if x.strip()]
    unknown = set(names) - set(CORE_DATASETS)
    if unknown:
        typer.echo(f"unknown datasets: {unknown}", err=True)
        raise typer.Exit(code=1)
    for ds in names:
        man = build_manifest(processed, ds, seed=seed)
        path = write_manifest(man, splits_root)
        typer.echo(f"[done] {ds} -> {path}")


if __name__ == "__main__":
    try:
        app()
    except KeyboardInterrupt:
        sys.exit(130)
