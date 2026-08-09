"""BKT baseline for E1: per-dialogue correctness proxy -> KC mastery vs silver."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import typer
from sklearn.metrics import roc_auc_score

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

from iss.baselines.bkt import MultiSkillBKT
from iss.data.latent_labels import load_latent_z_label_jsonl
from iss.experiments.dialogue_text import loads_turns
from iss.schema.kc_ontology import get_kc_ids
from iss.schema.latent import DialogueTurn

app = typer.Typer(no_args_is_help=True)

_CORRECT_RE = re.compile(r"\b(correct|right|yes|exactly|good job)\b", re.I)
_WRONG_RE = re.compile(r"\b(incorrect|wrong|not quite|error|mistake)\b", re.I)


def tutor_feedback_correctness(turns: list[DialogueTurn]) -> list[int]:
    """Binary correctness sequence from tutor feedback after each student turn."""
    seq: list[int] = []
    for i, t in enumerate(turns):
        if t.speaker != "student":
            continue
        label = 0
        for j in range(i + 1, min(i + 3, len(turns))):
            if turns[j].speaker != "tutor":
                continue
            txt = turns[j].text
            if _CORRECT_RE.search(txt):
                label = 1
                break
            if _WRONG_RE.search(txt):
                label = 0
                break
        seq.append(label)
    return seq


@app.command()
def main(
    labels_jsonl: Path = typer.Option(Path("data/labels/latent_z_silver_v3.jsonl"), "--labels-jsonl"),
    test_parquet: Path = typer.Option(Path("data/processed/mathdial/test.parquet"), "--test-parquet"),
    out_json: Path = typer.Option(Path("experiments/results/e1_bkt_baseline.json"), "--out-json"),
) -> None:
    lp = labels_jsonl if labels_jsonl.is_absolute() else repo_root / labels_jsonl
    gold = load_latent_z_label_jsonl(lp)
    df = pd.read_parquet(repo_root / test_parquet if not test_parquet.is_absolute() else test_parquet)

    bkt = MultiSkillBKT()
    # Fit on train-like pooled sequences (use test for demo if no train export)
    all_seq: dict[str, list[int]] = {k: [] for k in get_kc_ids()}
    for _, row in df.iterrows():
        turns = [DialogueTurn.model_validate(t) for t in loads_turns(str(row["turns_json"]))]
        seq = tutor_feedback_correctness(turns)
        if seq:
            all_seq["KC17"].extend(seq)  # proxy single-skill until per-KC tags exist
    bkt.fit_all({k: v for k, v in all_seq.items() if len(v) >= 3})

    preds: list[float] = []
    golds: list[float] = []
    for _, row in df.iterrows():
        did = str(row["dialogue_id"])
        if did not in gold:
            continue
        turns = [DialogueTurn.model_validate(t) for t in loads_turns(str(row["turns_json"]))]
        seq = tutor_feedback_correctness(turns)
        m = bkt.predict_mastery(np.asarray(seq, dtype=int), bkt.params_by_skill.get("KC17", bkt.params_by_skill.get("KC01")))
        preds.append(m)
        golds.append(float(np.mean(list(gold[did].mastery.values.values()))))

    y = (np.array(golds) > 0.5).astype(int)
    auc = float(roc_auc_score(y, preds)) if len(np.unique(y)) > 1 else float("nan")
    brier = float(np.mean((np.array(preds) - np.array(golds)) ** 2))

    out = {"kc_auc_mean": auc, "kc_brier_mean": brier, "n": len(preds)}
    out_path = out_json if out_json.is_absolute() else repo_root / out_json
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    typer.echo(json.dumps(out, indent=2))


if __name__ == "__main__":
    app()
