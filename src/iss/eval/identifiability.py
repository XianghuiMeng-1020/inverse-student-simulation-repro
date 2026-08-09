"""Identifiability proxies vs dialogue length (plan p9 scaffold)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pandas as pd

from iss.eval.metrics import mean_bernoulli_entropy
from iss.experiments.dialogue_text import loads_turns
from iss.forward.pseudo_z import pseudo_latent_z_from_prefix
from iss.schema.latent import DialogueTurn


def iter_mathdial_prefix_stats(
    parquet_path: Path,
    *,
    t_grid: tuple[int, ...] = (2, 4, 8, 16, 32),
    limit_rows: int = 0,
) -> Iterator[dict[str, float | int | str]]:
    df = pd.read_parquet(parquet_path, columns=["dialogue_id", "turns_json"])
    n = len(df) if limit_rows <= 0 else min(limit_rows, len(df))
    for i in range(n):
        row = df.iloc[i]
        turns = [DialogueTurn.model_validate(t) for t in loads_turns(str(row["turns_json"]))]
        if len(turns) < 2:
            continue
        for t_len in t_grid:
            if len(turns) < t_len:
                continue
            prefix = turns[:t_len]
            z = pseudo_latent_z_from_prefix(prefix)
            h = mean_bernoulli_entropy(list(z.mastery.values.values()))
            yield {
                "dialogue_id": str(row["dialogue_id"]),
                "T": t_len,
                "mean_mastery_entropy": h,
                "n_turns_total": len(turns),
            }


def write_identifiability_csv(
    parquet_path: Path,
    out_csv: Path,
    *,
    t_grid: tuple[int, ...] = (2, 4, 8, 16, 32),
    limit_rows: int = 0,
) -> None:
    rows = list(iter_mathdial_prefix_stats(parquet_path, t_grid=t_grid, limit_rows=limit_rows))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)


def aggregate_entropy_curve(csv_path: Path) -> dict[str, float]:
    df = pd.read_csv(csv_path)
    g = df.groupby("T")["mean_mastery_entropy"].mean()
    return {f"mean_entropy_T{int(k)}": float(v) for k, v in g.items()}


def write_aggregate_json(csv_path: Path, out_json: Path) -> None:
    out_json.write_text(json.dumps(aggregate_entropy_curve(csv_path), indent=2), encoding="utf-8")
