"""E7 Bridge closed-set with improved prompt, n=200."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

from iss.experiments.e7_openai import run_e7_openai_bridge_error_type

app = typer.Typer(no_args_is_help=True)


@app.command()
def main(
    n_samples: int = typer.Option(200, "--n-samples"),
    out_json: Path = typer.Option(Path("experiments/results/e7_bridge_v3.json"), "--out"),
) -> None:
    res = run_e7_openai_bridge_error_type(
        repo_root / "data" / "processed",
        n_samples=n_samples,
        seed=42,
        model="gpt-4o-mini",
        e_topk=25,
    )
    out_path = out_json if out_json.is_absolute() else repo_root / out_json
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, indent=2), encoding="utf-8")
    typer.echo(json.dumps({k: res[k] for k in res if k != "details"}, indent=2))


if __name__ == "__main__":
    app()
