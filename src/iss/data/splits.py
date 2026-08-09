"""Train/dev/test manifest generation (plan p4-12).

Stratification proxy: ``n_turns`` tertiles (KC coverage tags are not yet in all
processed shards).  Deterministic under ``seed`` via ``sklearn`` splitters.
Manifests are written to ``data/splits/`` (gitignored by default).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class SplitManifest:
    dataset: str
    seed: int
    train_ids: list[str]
    dev_ids: list[str]
    test_ids: list[str]
    stratify: str
    n_total: int

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["n_train"] = len(self.train_ids)
        d["n_dev"] = len(self.dev_ids)
        d["n_test"] = len(self.test_ids)
        return d


def _turn_bucket(series: pd.Series) -> np.ndarray:
    s = series.astype(float).values
    if len(s) == 0:
        return np.array([], dtype=int)
    nuniq = int(pd.Series(s).nunique())
    if nuniq <= 1:
        return np.zeros(len(s), dtype=int)
    try:
        b = pd.qcut(pd.Series(s), q=min(3, nuniq), labels=False, duplicates="drop")
        return b.astype(int).to_numpy()
    except ValueError:
        b = pd.cut(s, bins=min(3, max(2, nuniq)), labels=False, include_lowest=True)
        return pd.Series(b).fillna(0).astype(int).to_numpy()


def _collect_shard_paths(processed_root: Path, dataset: str) -> list[Path]:
    base = processed_root / dataset
    if not base.exists():
        msg = f"missing processed dir: {base}"
        raise FileNotFoundError(msg)
    if dataset == "mathdial":
        return [p for p in (base / "train.parquet", base / "test.parquet") if p.exists()]
    if dataset == "bridge":
        return [p for p in (base / "train.parquet", base / "validation.parquet", base / "test.parquet") if p.exists()]
    if dataset == "cima":
        return [base / "all.parquet"] if (base / "all.parquet").exists() else []
    if dataset == "talkmoves":
        keys = (
            "train_teacher.parquet",
            "train_student.parquet",
            "test_teacher.parquet",
            "test_student.parquet",
        )
        return [base / k for k in keys if (base / k).exists()]
    msg = f"unknown dataset key: {dataset}"
    raise ValueError(msg)


def load_concat_parquet(
    processed_root: Path,
    dataset: str,
    *,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    paths = _collect_shard_paths(processed_root, dataset)
    parts = [
        pd.read_parquet(p) if columns is None else pd.read_parquet(p, columns=columns) for p in paths
    ]
    return pd.concat(parts, ignore_index=True).drop_duplicates(subset=["dialogue_id"])


def load_dialogue_frame(processed_root: Path, dataset: str) -> pd.DataFrame:
    return load_concat_parquet(processed_root, dataset, columns=["dialogue_id", "n_turns"])


def build_manifest(
    processed_root: Path,
    dataset: str,
    *,
    seed: int,
    train_frac: float = 0.7,
    dev_frac_of_rest: float = 1.0 / 3.0,
) -> SplitManifest:
    """70/10/20 split: ``dev_frac_of_rest`` applied to the 30% holdout → 10%/20%."""

    df = load_dialogue_frame(processed_root, dataset)
    if len(df) < 6:
        msg = f"too few dialogues ({len(df)}) to stratify-split for {dataset}"
        raise ValueError(msg)
    strat = _turn_bucket(df["n_turns"])
    ids = df["dialogue_id"].astype(str).tolist()
    id_arr = np.array(ids)
    rest_frac = 1.0 - train_frac
    test_frac_of_rest = 1.0 - dev_frac_of_rest

    try:
        id_train, id_temp, _, strat_temp = train_test_split(
            id_arr,
            strat,
            test_size=rest_frac,
            random_state=seed,
            stratify=strat,
        )
    except ValueError:
        id_train, id_temp = train_test_split(
            id_arr,
            test_size=rest_frac,
            random_state=seed,
        )
        strat_temp = None

    if strat_temp is not None and len(np.unique(strat_temp)) >= 2:
        id_dev, id_test, _, _ = train_test_split(
            id_temp,
            strat_temp,
            test_size=test_frac_of_rest,
            random_state=seed + 1,
            stratify=strat_temp,
        )
    else:
        id_dev, id_test = train_test_split(
            id_temp,
            test_size=test_frac_of_rest,
            random_state=seed + 1,
        )
    return SplitManifest(
        dataset=dataset,
        seed=seed,
        train_ids=sorted(id_train.tolist()),
        dev_ids=sorted(id_dev.tolist()),
        test_ids=sorted(id_test.tolist()),
        stratify="n_turns_tertile",
        n_total=len(ids),
    )


def write_manifest(manifest: SplitManifest, splits_root: Path) -> Path:
    out_dir = splits_root / manifest.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(manifest.to_json(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def load_manifest(path: Path) -> SplitManifest:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return SplitManifest(
        dataset=str(raw["dataset"]),
        seed=int(raw["seed"]),
        train_ids=list(raw["train_ids"]),
        dev_ids=list(raw["dev_ids"]),
        test_ids=list(raw["test_ids"]),
        stratify=str(raw["stratify"]),
        n_total=int(raw["n_total"]),
    )


def manifest_fold_ids(manifest: SplitManifest, fold: str) -> frozenset[str]:
    """Return dialogue ids for ``fold`` (``train`` / ``dev`` / ``test``)."""

    f = fold.strip().lower()
    if f == "train":
        return frozenset(manifest.train_ids)
    if f == "dev":
        return frozenset(manifest.dev_ids)
    if f == "test":
        return frozenset(manifest.test_ids)
    msg = f"fold must be train|dev|test, got {fold!r}"
    raise ValueError(msg)
