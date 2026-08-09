"""E5 agreement: silver v3 vs v2 vs pseudo-Z on expert pool (no API)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import typer
from sklearn.metrics import cohen_kappa_score

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

from iss.data.latent_labels import load_latent_z_label_jsonl
from iss.forward.pseudo_z import pseudo_latent_z_from_prefix
from iss.experiments.dialogue_text import loads_turns
from iss.schema.kc_ontology import get_kc_ids
from iss.schema.latent import DialogueTurn

app = typer.Typer(no_args_is_help=True)


@app.command()
def main(
    pool_dir: Path = typer.Option(Path("data/labels/expert_pool_test"), "--pool-dir"),
    v3_jsonl: Path = typer.Option(Path("data/labels/latent_z_silver_v3.jsonl"), "--v3"),
    v2_jsonl: Path = typer.Option(Path("data/labels/latent_z_silver_v2.jsonl"), "--v2"),
    out_json: Path = typer.Option(Path("experiments/results/e5_expert_agreement.json"), "--out"),
) -> None:
    rr = repo_root.resolve()
    v3 = load_latent_z_label_jsonl(rr / v3_jsonl if not v3_jsonl.is_absolute() else v3_jsonl)
    v2 = load_latent_z_label_jsonl(rr / v2_jsonl if not v2_jsonl.is_absolute() else v2_jsonl)
    manifest = json.loads((rr / pool_dir / "manifest.json").read_text(encoding="utf-8"))

    kc_keys = get_kc_ids()
    ya, yb, yp = [], [], []
    mae_v3_v2 = []
    for did in manifest:
        p = rr / pool_dir / f"{did}.json"
        turns = [DialogueTurn.model_validate(t) for t in json.loads(p.read_text())["turns"]]
        if did not in v3 or did not in v2:
            continue
        za, zb, zp = v3[did], v2[did], pseudo_latent_z_from_prefix(turns)
        for k in kc_keys:
            ya.append(1 if za.mastery.values[k] > 0.5 else 0)
            yb.append(1 if zb.mastery.values[k] > 0.5 else 0)
            yp.append(1 if zp.mastery.values[k] > 0.5 else 0)
            mae_v3_v2.append(abs(za.mastery.values[k] - zb.mastery.values[k]))

    report = {
        "n_dialogues": len(manifest),
        "rater_a_model": "silver_v3",
        "rater_b_model": "silver_v2",
        "cohen_kappa_kc_binary": float(cohen_kappa_score(ya, yb)),
        "cohen_kappa_v3_vs_pseudo": float(cohen_kappa_score(ya, yp)),
        "silver_vs_rater_a_kc_mae": float(np.mean(mae_v3_v2)),
        "note": "Inter-label agreement on stratified test pool (v3 vs v2); supplements API dual-rater when available.",
    }
    out_path = out_json if out_json.is_absolute() else rr / out_json
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    typer.echo(json.dumps(report, indent=2))


if __name__ == "__main__":
    app()
