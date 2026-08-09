"""Validate silver v3 label statistics against overhaul targets."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import typer

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

from iss.schema.kc_ontology import get_kc_ids
from iss.schema.latent import LatentZ
from iss.schema.misconception_catalogue import get_misconception_ids

app = typer.Typer(no_args_is_help=True)


@app.command()
def main(
    labels_jsonl: Path = typer.Option(Path("data/labels/latent_z_silver_v3.jsonl"), "--labels-jsonl"),
    out_json: Path = typer.Option(Path("experiments/results/labels_v3_stats.json"), "--out-json"),
) -> None:
    path = labels_jsonl if labels_jsonl.is_absolute() else repo_root / labels_jsonl
    if not path.is_file():
        typer.echo(f"[err] missing {path}", err=True)
        raise typer.Exit(code=1)

    kc_all: list[float] = []
    misc_nonzero: list[int] = []
    metacog_dims: dict[str, list[float]] = {
        "monitoring_accuracy": [],
        "help_seeking_ratio": [],
        "confidence_correctness_gap": [],
        "hint_uptake": [],
    }
    spread_ok = 0
    n = 0

    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        z = LatentZ.model_validate(rec["latent"] if "latent" in rec else rec)

        vals = list(z.mastery.values.values())
        kc_all.extend(vals)
        hi = sum(1 for v in vals if v >= 0.7)
        lo = sum(1 for v in vals if v <= 0.3)
        if hi >= 3 and lo >= 2:
            spread_ok += 1

        probs = list(z.misconceptions.probs.values())
        misc_nonzero.append(sum(1 for p in probs if p >= 0.01))

        for k in metacog_dims:
            metacog_dims[k].append(float(getattr(z.metacog, k)))
        n += 1

    kc_std = float(np.std(kc_all)) if kc_all else 0.0
    misc_frac = float(np.mean([x > 0 for x in misc_nonzero])) if misc_nonzero else 0.0
    meta_std = {k: float(np.std(v)) for k, v in metacog_dims.items()}

    stats = {
        "n_rows": n,
        "kc_std": kc_std,
        "kc_std_target": 0.15,
        "kc_std_pass": kc_std > 0.15,
        "misc_nonzero_frac": misc_frac,
        "misc_nonzero_target": 0.35,
        "misc_nonzero_pass": misc_frac > 0.35,
        "metacog_std": meta_std,
        "metacog_std_pass": all(v > 0.10 for v in meta_std.values()),
        "kc_spread_pass_frac": spread_ok / n if n else 0.0,
    }

    out_path = out_json if out_json.is_absolute() else repo_root / out_json
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    typer.echo(json.dumps(stats, indent=2))
    if not all([stats["kc_std_pass"], stats["misc_nonzero_pass"], stats["metacog_std_pass"]]):
        typer.echo("[warn] some v3 targets not met — review prompt or re-label", err=True)
        raise typer.Exit(code=2)
    typer.echo("[ok] v3 label statistics pass targets")


if __name__ == "__main__":
    app()
