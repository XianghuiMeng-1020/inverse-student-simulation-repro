"""Export identifiability proxy CSV (plan p9-03)."""

from __future__ import annotations

import sys
from pathlib import Path

import typer

from iss.eval.identifiability import write_aggregate_json, write_identifiability_csv

app = typer.Typer(no_args_is_help=True)


@app.command()
def main(
    repo_root: Path = typer.Option(Path("."), "--repo-root"),
    parquet: Path = typer.Option(Path("data/processed/mathdial/train.parquet"), "--parquet"),
    limit_rows: int = typer.Option(400, "--limit-rows"),
    out_csv: Path = typer.Option(
        Path("experiments/results/identifiability/mathdial_prefix_curve.csv"),
        "--out-csv",
    ),
) -> None:
    p = (repo_root / parquet).resolve() if not parquet.is_absolute() else parquet
    out = (repo_root / out_csv).resolve() if not out_csv.is_absolute() else out_csv
    write_identifiability_csv(p, out, limit_rows=limit_rows)
    agg = out.with_suffix(".summary.json")
    write_aggregate_json(out, agg)
    typer.echo(f"[done] wrote {out} and {agg}")


if __name__ == "__main__":
    try:
        app()
    except KeyboardInterrupt:
        sys.exit(130)
