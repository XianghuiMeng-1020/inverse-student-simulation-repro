"""E3 identifiability curve using trained inverter predictions (not pseudo-Z)."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Must run before torch is imported (via iss.*).
if "--cpu-only" in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import pandas as pd

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

from iss.eval.metrics import mean_bernoulli_entropy
from iss.experiments.dialogue_text import loads_turns
from iss.inverter.model import JointInverter
from iss.schema.latent import DialogueTurn
from hydra import compose, initialize_config_dir
from omegaconf import open_dict

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main(
    adapter_dir: Path,
    parquet: Path,
    t_grid: str,
    limit: int,
    max_gpu_gb: float,
    max_output_tokens: int,
    cpu_only: bool,
    ckpt_every: int,
) -> None:
    if cpu_only:
        log.info("CUDA disabled for CPU-only E3")

    cfg_dir = str((repo_root / "configs").resolve())
    with initialize_config_dir(config_dir=cfg_dir, version_base=None):
        cfg = compose(config_name="config", overrides=["model=inverter_3b"])
    with open_dict(cfg.model):
        cfg.model.max_gpu_gb = float(max_gpu_gb)
        cfg.model.max_output_tokens = int(max_output_tokens)
        cfg.model.force_cpu = cpu_only
        # Disable 4-bit quantization for inference: bitsandbytes 4-bit GPU kernel
        # is not stable on RTX 5090 (Blackwell sm_100). Use fp16 instead (~6 GB).
        cfg.model.load_in_4bit = False

    ad = adapter_dir if adapter_dir.is_absolute() else repo_root / adapter_dir
    log.info("Loading inverter adapter=%s cpu_only=%s", ad, cpu_only)
    inv = JointInverter(cfg, adapter_dir=str(ad), force_cpu=cpu_only)

    ks = [int(x) for x in t_grid.split(",") if x.strip()]
    pq = parquet if parquet.is_absolute() else repo_root / parquet
    df = pd.read_parquet(pq)
    n = min(limit, len(df)) if limit > 0 else len(df)
    log.info("Running E3 on %d dialogues, T_grid=%s", n, ks)

    out_dir = repo_root / "experiments" / "results" / "identifiability"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "mathdial_prefix_curve_inverter.csv"
    partial_path = out_dir / "mathdial_prefix_curve_inverter.partial.csv"

    rows: list[dict] = []
    if partial_path.exists():
        prev = pd.read_csv(partial_path)
        rows = prev.to_dict(orient="records")
        log.info("Resumed %d partial rows from %s", len(rows), partial_path)

    done_ids = {r["dialogue_id"] for r in rows}
    for i in range(n):
        did = str(df.iloc[i]["dialogue_id"])
        if did in done_ids:
            continue
        turns = [DialogueTurn.model_validate(t) for t in loads_turns(str(df.iloc[i]["turns_json"]))]
        # Collect all valid prefixes for this dialogue, then generate in one batch.
        prefixes: list[list[DialogueTurn]] = []
        t_labels: list[str] = []
        for t_len in ks:
            if t_len > 0 and len(turns) < t_len:
                continue
            prefixes.append(turns if t_len <= 0 else turns[:t_len])
            t_labels.append("full" if t_len <= 0 else str(t_len))

        zs = inv.generate_z_batch(prefixes)
        for label, z in zip(t_labels, zs):
            h = mean_bernoulli_entropy(list(z.mastery.values.values()))
            rows.append({"dialogue_id": did, "T": label, "mean_mastery_entropy": h})

        done_ids.add(did)
        log.info("dialogue %d/%d id=%s rows=%d", i + 1, n, did, len(rows))
        if (i + 1) % ckpt_every == 0 or (i + 1) == n:
            pd.DataFrame(rows).to_csv(partial_path, index=False)
            log.info("checkpoint %d/%d dialogues (%d rows)", i + 1, n, len(rows))

    pd.DataFrame(rows).to_csv(csv_path, index=False)
    partial_path.unlink(missing_ok=True)
    rows_df = pd.DataFrame(rows)
    agg = rows_df.groupby("T")["mean_mastery_entropy"].mean().to_dict() if rows_df.size else {}
    summary = {f"mean_entropy_T{k}": float(v) for k, v in agg.items()}
    (out_dir / "mathdial_prefix_curve_inverter.summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    log.info("[done] %s", summary)
    lock = repo_root / "experiments" / "logs" / "e3_v3.lock"
    lock.unlink(missing_ok=True)


if __name__ == "__main__":
    lock_path = Path(__file__).resolve().parents[1] / "experiments" / "logs" / "e3_v3.lock"
    p = argparse.ArgumentParser(description="E3 identifiability curve")
    p.add_argument("--adapter-dir", default="experiments/checkpoints/inverter_3b_qlora_silver_v3_s42/inverter_lora")
    p.add_argument("--parquet", default="data/processed/mathdial/train.parquet")
    p.add_argument("--t-grid", default="2,4,8,0")
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--max-gpu-gb", type=float, default=8.0)
    p.add_argument("--max-output-tokens", type=int, default=1536)
    p.add_argument("--cpu-only", action="store_true")
    p.add_argument("--ckpt-every", type=int, default=20)
    args = p.parse_args()
    try:
        main(
            adapter_dir=Path(args.adapter_dir),
            parquet=Path(args.parquet),
            t_grid=args.t_grid,
            limit=args.limit,
            max_gpu_gb=args.max_gpu_gb,
            max_output_tokens=args.max_output_tokens,
            cpu_only=args.cpu_only,
            ckpt_every=args.ckpt_every,
        )
        sys.exit(0)
    except Exception:
        lock_path.unlink(missing_ok=True)
        log.exception("E3 identifiability failed")
        sys.exit(1)
