"""Build unified parquet shards under ``data/processed/<dataset>/``."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import typer
from tqdm import tqdm

from iss.data.bridge import row_to_dialogue as bridge_row
from iss.data.cima import iter_cima_dialogues
from iss.data.mathdial import load_mathdial_hf
from iss.data.mathdial import row_to_dialogue as mathdial_row
from iss.data.registry import CORE_DATASETS
from iss.data.talkmoves import default_talkmoves_paths, iter_talkmoves_dialogues
from iss.schema.latent import Dialogue

app = typer.Typer(no_args_is_help=True)


def _dialogue_row(d: Dialogue) -> dict:
    return {
        "dialogue_id": d.dialogue_id,
        "dataset": d.metadata.get("dataset", ""),
        "language": d.language,
        "n_turns": len(d.turns),
        "turns_json": json.dumps([t.model_dump(mode="json") for t in d.turns], ensure_ascii=False),
        "metadata_json": json.dumps(d.metadata, ensure_ascii=False),
    }


def _write_parquet(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out_path, index=False)


def _collect_mathdial(repo_root: Path, limit: int) -> None:
    ds = load_mathdial_hf()
    for split in ds:
        rows_out: list[dict] = []
        part = ds[split]
        n = len(part) if limit <= 0 else min(limit, len(part))
        for i in tqdm(range(n), desc=f"mathdial/{split}"):
            d = mathdial_row(part[i], split=str(split), idx=i)
            if d is not None:
                rows_out.append(_dialogue_row(d))
        _write_parquet(rows_out, repo_root / "data" / "processed" / "mathdial" / f"{split}.parquet")


def _collect_bridge(repo_root: Path, limit: int) -> None:
    from datasets import load_dataset

    ds = load_dataset("rose-e-wang/bridge")
    for split in ds:
        rows_out: list[dict] = []
        part = ds[split]
        n = len(part) if limit <= 0 else min(limit, len(part))
        for i in tqdm(range(n), desc=f"bridge/{split}"):
            d = bridge_row(part[i], split=str(split), idx=i)
            if d is not None:
                rows_out.append(_dialogue_row(d))
        _write_parquet(rows_out, repo_root / "data" / "processed" / "bridge" / f"{split}.parquet")


def _collect_cima(repo_root: Path, limit: int) -> None:
    path = repo_root / "data" / "raw" / "cima" / "dataset.json"
    if not path.exists():
        typer.echo(f"[warn] CIMA not found at {path}; run scripts/download_data.py", err=True)
        return
    rows_out: list[dict] = []
    for j, d in enumerate(
        tqdm(iter_cima_dialogues(dataset_path=path), desc="cima/all", total=None),
    ):
        if limit > 0 and j >= limit:
            break
        rows_out.append(_dialogue_row(d))
    _write_parquet(rows_out, repo_root / "data" / "processed" / "cima" / "all.parquet")


def _collect_talkmoves(repo_root: Path, limit: int) -> None:
    paths = default_talkmoves_paths(repo_root)
    for key, tsv in paths.items():
        if not tsv.exists():
            typer.echo(f"[warn] missing {tsv}", err=True)
            continue
        _, role = key.split("_", 1)
        rows_out: list[dict] = []
        for j, d in enumerate(
            tqdm(
                iter_talkmoves_dialogues(tsv_path=tsv, split_name=key, role=role),
                desc=f"talkmoves/{key}",
                total=None,
            ),
        ):
            if limit > 0 and j >= limit:
                break
            rows_out.append(_dialogue_row(d))
        out = repo_root / "data" / "processed" / "talkmoves" / f"{key}.parquet"
        _write_parquet(rows_out, out)


@app.command()
def main(
    repo_root: Path = typer.Option(Path("."), "--repo-root"),
    datasets: str = typer.Option(
        ",".join(CORE_DATASETS),
        "--datasets",
        help=f"Comma list from {CORE_DATASETS}",
    ),
    limit: int = typer.Option(
        0,
        "--limit",
        help="Max dialogues per shard (0 = all). Use e.g. 200 for smoke builds.",
    ),
) -> None:
    names = {x.strip() for x in datasets.split(",") if x.strip()}
    unknown = names - set(CORE_DATASETS)
    if unknown:
        typer.echo(f"Unknown datasets: {unknown}", err=True)
        raise typer.Exit(code=1)
    if "mathdial" in names:
        _collect_mathdial(repo_root, limit)
    if "bridge" in names:
        _collect_bridge(repo_root, limit)
    if "cima" in names:
        _collect_cima(repo_root, limit)
    if "talkmoves" in names:
        _collect_talkmoves(repo_root, limit)
    typer.echo("[done] processed shards written under data/processed/")


if __name__ == "__main__":
    try:
        app()
    except KeyboardInterrupt:
        sys.exit(130)
