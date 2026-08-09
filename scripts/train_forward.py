"""LoRA SFT for the forward student simulator ``F(Z, \\text{ctx})``.

Run (from repo root; optional ``--manifest`` aligns rows with ``scripts/build_splits.py``)::

    uv run python scripts/build_forward_sft_jsonl.py --repo-root .
    uv run python scripts/train_forward.py model=forward_3b \\
        ++paths.forward_sft_jsonl=data/forward_sft/mathdial_forward_train.jsonl
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
    if str(cfg.model.name) != "forward_3b":
        raise SystemExit("Pass model=forward_3b for forward training.")

    jsonl = Path(str(cfg.paths.forward_sft_jsonl))
    if not jsonl.is_file():
        raise SystemExit(
            f"Missing JSONL: {jsonl}. Build with:\n"
            "  uv run python scripts/build_forward_sft_jsonl.py --repo-root .",
        )
    records = [
        json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    cap = int(cfg.pipeline.max_train_records)
    if cap > 0:
        records = records[:cap]

    out_dir = Path(str(cfg.paths.checkpoints)) / "forward_lora"
    out_dir.mkdir(parents=True, exist_ok=True)
    max_len = int(cfg.model.max_input_tokens)
    train_causal_lm_sft(cfg=cfg, records=records, output_dir=str(out_dir), max_length=max_len)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
