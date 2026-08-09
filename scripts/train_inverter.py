"""LoRA SFT for the joint inverter (dialogue -> ``LatentZ`` JSON).

Run (optional ``--manifest`` / ``--manifest-fold`` match ``data/splits/``)::

    uv run python scripts/build_inverter_sft_jsonl.py --repo-root .
    uv run python scripts/train_inverter.py model=inverter_3b \\
        ++paths.inverter_sft_jsonl=data/inverter_sft/mathdial_inverter_train.jsonl
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from iss.training.causal_sft import train_causal_lm_sft


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    if str(cfg.model.name) != "inverter_3b":
        raise SystemExit("Pass model=inverter_3b for inverter training.")

    jsonl = Path(str(cfg.paths.inverter_sft_jsonl))
    if not jsonl.is_file():
        raise SystemExit(
            f"Missing JSONL: {jsonl}. Build with:\n"
            "  uv run python scripts/build_inverter_sft_jsonl.py --repo-root .",
        )
    records = [
        json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    cap = int(cfg.pipeline.max_train_records)
    if cap > 0:
        records = records[:cap]

    out_dir = Path(str(cfg.paths.checkpoints)) / "inverter_lora"
    out_dir.mkdir(parents=True, exist_ok=True)
    max_len = int(cfg.model.max_input_tokens) + int(cfg.model.max_output_tokens)
    train_causal_lm_sft(cfg=cfg, records=records, output_dir=str(out_dir), max_length=max_len)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
