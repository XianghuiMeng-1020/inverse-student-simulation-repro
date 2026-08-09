"""Emit simple markdown stats for processed parquet shards."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import typer

app = typer.Typer(no_args_is_help=True)


@app.command()
def main(
    repo_root: Path = typer.Option(Path("."), "--repo-root"),
    out: Path = typer.Option(Path("data/stats/datasets.md"), "--out"),
) -> None:
    root = (repo_root / "data" / "processed").resolve()
    if not root.exists():
        typer.echo(f"[err] missing {root}", err=True)
        raise typer.Exit(code=1)
    lines: list[str] = ["# Processed dataset shards", ""]
    for path in sorted(root.rglob("*.parquet")):
        df = pd.read_parquet(path)
        rel = path.relative_to(root)
        lines.append(f"## `{rel}`")
        lines.append(f"- rows: **{len(df)}**")
        if "n_turns" in df.columns:
            lines.append(f"- mean n_turns: **{df['n_turns'].mean():.2f}**")
        lines.append("")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    typer.echo(f"[done] wrote {out}")


if __name__ == "__main__":
    try:
        app()
    except KeyboardInterrupt:
        sys.exit(130)
