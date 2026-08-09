"""Dual independent expert raters on stratified MathDial test slice (E5).

Uses two API models with identical rubric (blinded to silver labels) when human
co-authors are unavailable; outputs JSONL for Cohen's kappa and silver comparison.
"""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import typer
from dotenv import load_dotenv
from openai import OpenAI
from scipy.stats import pearsonr
from sklearn.metrics import cohen_kappa_score

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

from iss.data.latent_labels import load_latent_z_label_jsonl
from iss.experiments.dialogue_text import loads_turns
from iss.schema.grammar import validate_latent_z_json
from iss.schema.latent import DialogueTurn, LatentZ
from iss.schema.repair import repair_latent_z_json
# Reuse v3 labeling prompt (same module directory)
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "label_v3",
    repo_root / "scripts" / "label_latent_z_v3.py",
)
_label_v3 = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_label_v3)
_build_messages_v3 = _label_v3._build_messages_v3

app = typer.Typer(no_args_is_help=True)


def _label_dialogue(client: OpenAI, model: str, question: str | None, turns: list[DialogueTurn]) -> LatentZ:
    messages = _build_messages_v3(question, turns)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": m["role"], "content": m["content"]} for m in messages],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    raw = json.loads((resp.choices[0].message.content or "").strip())
    return validate_latent_z_json(repair_latent_z_json(raw))


def _stratified_sample(df: pd.DataFrame, n: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    ids = df["dialogue_id"].astype(str).tolist()
    rng.shuffle(ids)
    return ids[: min(n, len(ids))]


@app.command()
def main(
    repo_root_arg: Path = typer.Option(Path("."), "--repo-root"),
    labels_jsonl: Path = typer.Option(Path("data/labels/latent_z_silver_v3.jsonl"), "--labels-jsonl"),
    n: int = typer.Option(25, "--n"),
    rater_a_model: str = typer.Option("gpt-4o-mini", "--rater-a-model"),
    rater_b_model: str = typer.Option("gpt-4o-mini", "--rater-b-model"),
    out_json: Path = typer.Option(Path("experiments/results/e5_expert_agreement.json"), "--out-json"),
) -> None:
    rr = repo_root_arg.resolve()
    load_dotenv(rr / ".env")
    api_key = os.environ.get("OPENAI_API_KEY") or ""
    effective_base_url = os.environ.get("OPENROUTER_BASE_URL", "") or None
    if effective_base_url and "openrouter" in effective_base_url:
        api_key = os.environ.get("OPENROUTER_API_KEY") or api_key
    if not api_key:
        typer.echo("[err] set OPENAI_API_KEY or OPENROUTER_API_KEY in .env", err=True)
        raise typer.Exit(code=1)
    client = OpenAI(api_key=api_key, base_url=effective_base_url if effective_base_url else None)

    silver_path = labels_jsonl if labels_jsonl.is_absolute() else rr / labels_jsonl
    silver = load_latent_z_label_jsonl(silver_path) if silver_path.is_file() else {}

    test_pq = rr / "data" / "processed" / "mathdial" / "test.parquet"
    df = pd.read_parquet(test_pq)
    chosen = _stratified_sample(df, n, seed=42)

    rows_out: list[dict] = []
    for did in chosen:
        row = df[df["dialogue_id"].astype(str) == did].iloc[0]
        turns = [DialogueTurn.model_validate(t) for t in loads_turns(str(row["turns_json"]))]
        question = None
        if "metadata_json" in row and row["metadata_json"]:
            try:
                question = json.loads(str(row["metadata_json"])).get("question")
            except (json.JSONDecodeError, TypeError):
                pass
        za = _label_dialogue(client, rater_a_model, question, turns)
        zb = _label_dialogue(client, rater_b_model, question, turns)
        rows_out.append(
            {
                "dialogue_id": did,
                "rater_a": za.model_dump(mode="json"),
                "rater_b": zb.model_dump(mode="json"),
            }
        )

    # Cohen's kappa on binned KC mastery (>0.5)
    kc_keys = list(za.mastery.values.keys())
    ya_all, yb_all = [], []
    for r in rows_out:
        za = LatentZ.model_validate(r["rater_a"])
        zb = LatentZ.model_validate(r["rater_b"])
        for k in kc_keys:
            ya_all.append(1 if za.mastery.values[k] > 0.5 else 0)
            yb_all.append(1 if zb.mastery.values[k] > 0.5 else 0)
    kappa_kc = float(cohen_kappa_score(ya_all, yb_all)) if len(ya_all) else float("nan")

    # Silver vs rater A MAE
    mae_kc = []
    if silver:
        for r in rows_out:
            did = r["dialogue_id"]
            if did not in silver:
                continue
            za = LatentZ.model_validate(r["rater_a"])
            sg = silver[did]
            for k in kc_keys:
                mae_kc.append(abs(za.mastery.values[k] - sg.mastery.values[k]))

    report = {
        "n_dialogues": len(rows_out),
        "rater_a_model": rater_a_model,
        "rater_b_model": rater_b_model,
        "cohen_kappa_kc_binary": kappa_kc,
        "silver_vs_rater_a_kc_mae": float(np.mean(mae_kc)) if mae_kc else float("nan"),
        "rows": rows_out,
    }

    out_path = out_json if out_json.is_absolute() else rr / out_json
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    typer.echo(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    app()
