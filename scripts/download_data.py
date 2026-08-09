"""Download / verify the four public raw corpora used in ISS.

HF datasets (MathDial, Bridge) populate the Hugging Face cache on first access.
CIMA + TalkMoves are shallow-cloned into ``data/raw/`` (gitignored).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import typer

app = typer.Typer(no_args_is_help=True)

CLONES: tuple[tuple[str, str], ...] = (
    ("cima", "https://github.com/kstats/CIMA.git"),
    ("talkmoves", "https://github.com/SumnerLab/TalkMoves.git"),
)


@app.command()
def main(
    repo_root: Path = typer.Option(Path("."), help="Repository root (contains data/)."),
) -> None:
    raw = (repo_root / "data" / "raw").resolve()
    raw.mkdir(parents=True, exist_ok=True)
    for folder, url in CLONES:
        target = raw / folder
        if target.exists():
            typer.echo(f"[skip] {target} already exists")
            continue
        typer.echo(f"[git] cloning {url} -> {target}")
        subprocess.run(
            ["git", "clone", "--depth", "1", url, str(target)],
            check=True,
        )
    typer.echo(
        "[hf] MathDial (eth-nlped/mathdial) and Bridge (rose-e-wang/bridge) "
        "download automatically on first `datasets.load_dataset` call."
    )


if __name__ == "__main__":
    try:
        app()
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)
