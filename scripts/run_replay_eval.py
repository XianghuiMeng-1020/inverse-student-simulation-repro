"""Counterfactual replay (plan p10): smoke or E4 grid on a parquet shard.

Requires a trained forward LoRA adapter. If missing, writes a skip JSON.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import typer
from hydra import compose, initialize_config_dir

app = typer.Typer(no_args_is_help=True)


def _stable_seed(dialogue_id: str, anchor: int, horizon: int, salt: str) -> int:
    h = hashlib.sha256(f"{dialogue_id}|{anchor}|{horizon}|{salt}".encode()).hexdigest()
    return int(h[:8], 16)


@app.command()
def main(
    repo_root: Path = typer.Option(Path("."), "--repo-root"),
    adapter: Path | None = typer.Option(None, "--adapter"),
    parquet: Path = typer.Option(Path("data/processed/mathdial/train.parquet"), "--parquet"),
    limit: int = typer.Option(3, "--limit"),
    out: Path = typer.Option(Path("experiments/results/replay_smoke.json"), "--out"),
    max_gpu_gb: float = typer.Option(
        0.0,
        "--max-gpu-gb",
        help="Cap GPU memory (GiB) for the forward model and load in 4-bit. 0 = no cap (use all GPU).",
    ),
    e4_grid: bool = typer.Option(
        False,
        "--e4-grid/--smoke",
        help="Full E4: all student anchors, horizons, pseudo vs random-Z; else legacy 3-row smoke.",
    ),
    horizons: str = typer.Option(
        "1,3,5",
        "--horizons",
        help="Comma-separated student-turn horizons (E4 only).",
    ),
    random_z: bool = typer.Option(True, "--random-z/--no-random-z", help="E4: include uniform random Z' control."),
    silver_labels: Path | None = typer.Option(
        None,
        "--silver-labels",
        help="Optional JSONL of full-dialogue silver Z (same Z for all anchors in a dialogue).",
    ),
) -> None:
    adapter_path = adapter
    if adapter_path is None:
        ckpt_root = repo_root / "experiments" / "checkpoints"
        candidates = sorted(
            ckpt_root.glob("*/forward_lora"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        adapter_path = candidates[0] if candidates else None

    out_path = repo_root / out if not out.is_absolute() else out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if adapter_path is None or not adapter_path.exists():
        out_path.write_text(
            json.dumps({"status": "skipped", "reason": "no forward adapter found"}, indent=2),
            encoding="utf-8",
        )
        typer.echo(f"[skip] wrote {out_path}")
        raise typer.Exit(code=0)

    import pandas as pd
    from tqdm import tqdm

    from iss.data.latent_labels import load_latent_z_label_jsonl
    from iss.eval.replay import counterfactual_replay_row, mean_nll_next_k_student_turns
    from iss.experiments.dialogue_text import loads_turns
    from iss.forward.pseudo_z import pseudo_latent_z_from_prefix, uniform_random_latent_z
    from iss.forward.simulator import ForwardSimulator
    from iss.schema.latent import DialogueTurn

    cfg_dir = str((repo_root / "configs").resolve())
    with initialize_config_dir(config_dir=cfg_dir, version_base=None):
        cfg = compose(config_name="config", overrides=["model=forward_3b"])

    sim = ForwardSimulator.from_checkpoint(
        cfg,
        adapter_dir=str(adapter_path),
        max_gpu_gb=max_gpu_gb if max_gpu_gb > 0 else None,
    )
    pq = repo_root / parquet if not parquet.is_absolute() else parquet
    df = pd.read_parquet(pq)

    silver_map = None
    if silver_labels is not None:
        lp = repo_root / silver_labels if not silver_labels.is_absolute() else silver_labels
        if lp.is_file():
            silver_map = load_latent_z_label_jsonl(lp)

    max_len = int(cfg.model.max_input_tokens)
    rows_out: list[dict[str, object]] = []

    if not e4_grid:
        for i in range(min(limit, len(df))):
            turns_raw = loads_turns(str(df.iloc[i]["turns_json"]))
            turns = [DialogueTurn.model_validate(t) for t in turns_raw]
            stu_idx = next((j for j, t in enumerate(turns) if t.speaker == "student"), None)
            if stu_idx is None or stu_idx == 0:
                continue
            prefix = turns[:stu_idx]
            gold = turns[stu_idx].text
            z = pseudo_latent_z_from_prefix(prefix)
            row = counterfactual_replay_row(
                sim.model,
                sim.tokenizer,
                z=z,
                prefix_turns=prefix,
                gold_next_student=gold,
                max_length=max_len,
            )
            row["dialogue_id"] = str(df.iloc[i]["dialogue_id"])
            rows_out.append(row)
        payload = {"status": "ok", "mode": "smoke", "rows": rows_out}
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        typer.echo(f"[done] wrote {out_path}")
        return

    ks = [int(x.strip()) for x in horizons.split(",") if x.strip()]
    n_dialogues = len(df) if limit <= 0 else min(limit, len(df))

    for i in tqdm(range(n_dialogues), desc="e4_replay"):
        turns_raw = loads_turns(str(df.iloc[i]["turns_json"]))
        turns = [DialogueTurn.model_validate(t) for t in turns_raw]
        did = str(df.iloc[i]["dialogue_id"])
        stud_ix = [j for j, t in enumerate(turns) if t.speaker == "student"]
        z_full_silver = silver_map.get(did) if silver_map else None
        for stu_idx in stud_ix:
            if stu_idx == 0:
                continue
            prefix_for_z = turns[:stu_idx]
            z_pseudo = pseudo_latent_z_from_prefix(prefix_for_z)
            for h in ks:
                m_pseudo = mean_nll_next_k_student_turns(
                    sim.model,
                    sim.tokenizer,
                    z=z_pseudo,
                    turns=turns,
                    start_student_index=stu_idx,
                    horizon=h,
                    max_length=max_len,
                )
                rec: dict[str, object] = {
                    "dialogue_id": did,
                    "anchor_student_ix": stu_idx,
                    "horizon": h,
                    "nll_mean_pseudo": m_pseudo,
                }
                if random_z:
                    seed = _stable_seed(did, stu_idx, h, "randz")
                    z_r = uniform_random_latent_z(seed=seed)
                    m_r = mean_nll_next_k_student_turns(
                        sim.model,
                        sim.tokenizer,
                        z=z_r,
                        turns=turns,
                        start_student_index=stu_idx,
                        horizon=h,
                        max_length=max_len,
                    )
                    rec["nll_mean_random_z"] = m_r
                if z_full_silver is not None:
                    m_s = mean_nll_next_k_student_turns(
                        sim.model,
                        sim.tokenizer,
                        z=z_full_silver,
                        turns=turns,
                        start_student_index=stu_idx,
                        horizon=h,
                        max_length=max_len,
                    )
                    rec["nll_mean_silver_full_dialogue_z"] = m_s
                rows_out.append(rec)

    def _mean(xs: list[float]) -> float | None:
        ys = [x for x in xs if x is not None and not math.isnan(x) and math.isfinite(x)]
        return sum(ys) / len(ys) if ys else None

    pseudo_by_h: dict[int, list[float]] = {h: [] for h in ks}
    rand_by_h: dict[int, list[float]] = {h: [] for h in ks}
    silv_by_h: dict[int, list[float]] = {h: [] for h in ks}
    for r in rows_out:
        h = int(r["horizon"])
        v = r.get("nll_mean_pseudo")
        if isinstance(v, (int, float)) and v is not None:
            pseudo_by_h[h].append(float(v))
        v2 = r.get("nll_mean_random_z")
        if isinstance(v2, (int, float)) and v2 is not None:
            rand_by_h[h].append(float(v2))
        v3 = r.get("nll_mean_silver_full_dialogue_z")
        if isinstance(v3, (int, float)) and v3 is not None:
            silv_by_h[h].append(float(v3))

    summary = {
        "n_rows": len(rows_out),
        "horizons": ks,
        "mean_nll_pseudo": {str(h): _mean(pseudo_by_h[h]) for h in ks},
        "mean_nll_random_z": {str(h): _mean(rand_by_h[h]) for h in ks} if random_z else {},
        "mean_nll_silver": {str(h): _mean(silv_by_h[h]) for h in ks} if silver_map else {},
    }
    payload = {
        "status": "ok",
        "mode": "e4_grid",
        "parquet": str(parquet),
        "adapter": str(adapter_path),
        "summary": summary,
        "rows": rows_out,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    typer.echo(f"[done] wrote {out_path} n_rows={len(rows_out)}")


if __name__ == "__main__":
    try:
        app()
    except KeyboardInterrupt:
        sys.exit(130)
