"""Project-wide logging + W&B initialization.

W&B defaults to offline mode (set WANDB_MODE=offline in .env) for reviewer
reproducibility. Switch to online by exporting WANDB_API_KEY and
WANDB_MODE=online in the local environment.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger


def init_env(env_file: str | Path = ".env") -> None:
    """Load .env if present. Idempotent; safe to call multiple times."""
    env_path = Path(env_file)
    if env_path.exists():
        load_dotenv(env_path, override=False)


def init_logger(log_file: str | Path | None = None, level: str = "INFO") -> None:
    """Set up loguru with rich formatting and optional file sink."""
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
        colorize=True,
    )
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(log_path, level=level, rotation="100 MB", retention=5)


def init_wandb(
    project: str = "inverse-student-sim",
    run_name: str | None = None,
    config: dict | None = None,
    tags: list[str] | None = None,
    group: str | None = None,
):
    """Initialize W&B run. Returns the run object or None if disabled."""
    try:
        import wandb
    except ImportError:
        logger.warning("wandb not installed; skipping initialization.")
        return None

    mode = os.environ.get("WANDB_MODE", "offline")
    if mode == "disabled":
        logger.info("WANDB_MODE=disabled; skipping W&B init.")
        return None

    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", project),
        entity=os.environ.get("WANDB_ENTITY") or None,
        name=run_name,
        config=config,
        tags=tags,
        group=group,
        mode=mode,
        reinit="finish_previous",
    )
    logger.info(f"W&B initialized: mode={mode} project={project} run={run.name}")
    return run
