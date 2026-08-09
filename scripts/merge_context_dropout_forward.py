"""Merge standard forward SFT + oracle (Z-only) rows for context-dropout training."""

from __future__ import annotations

import json
import random
from pathlib import Path

import typer

app = typer.Typer(no_args_is_help=True)


@app.command()
def main(
    repo_root: Path = typer.Option(Path("."), "--repo-root"),
    forward_jsonl: Path = typer.Option(
        Path("data/forward_sft/mathdial_forward_train_silver_v3.jsonl"),
        "--forward-jsonl",
    ),
    oracle_jsonl: Path = typer.Option(
        Path("data/forward_sft/mathdial_oracle_forward_train.jsonl"),
        "--oracle-jsonl",
    ),
    dropout_frac: float = typer.Option(0.5, "--dropout-frac"),
    seed: int = typer.Option(42, "--seed"),
    out_name: str = typer.Option("mathdial_forward_context_dropout.jsonl", "--out-name"),
) -> None:
    rr = repo_root.resolve()
    fwd_path = rr / forward_jsonl if not forward_jsonl.is_absolute() else forward_jsonl
    orc_path = rr / oracle_jsonl if not oracle_jsonl.is_absolute() else oracle_jsonl
    rng = random.Random(seed)

    fwd_rows = [json.loads(l) for l in fwd_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    orc_rows = [json.loads(l) for l in orc_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    rng.shuffle(orc_rows)

    out_path = rr / "data" / "forward_sft" / out_name
    n_orc = 0
    with out_path.open("w", encoding="utf-8") as f:
        orc_i = 0
        for rec in fwd_rows:
            if rng.random() < dropout_frac and orc_i < len(orc_rows):
                o = orc_rows[orc_i]
                o.setdefault("meta", {})["mode"] = "context_dropout"
                f.write(json.dumps(o, ensure_ascii=False) + "\n")
                orc_i += 1
                n_orc += 1
            else:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    typer.echo(f"[done] wrote {len(fwd_rows)} base + {n_orc} dropout -> {out_path}")


if __name__ == "__main__":
    app()
