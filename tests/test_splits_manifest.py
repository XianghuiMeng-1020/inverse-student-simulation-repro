"""Tests for deterministic split manifests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from iss.data.splits import SplitManifest, build_manifest, manifest_fold_ids, write_manifest


def test_build_manifest_roundtrip(tmp_path: Path) -> None:
    proc = tmp_path / "processed" / "mathdial"
    proc.mkdir(parents=True)
    df = pd.DataFrame(
        {
            "dialogue_id": [f"d{i}" for i in range(30)],
            "n_turns": [2 + (i % 5) for i in range(30)],
        },
    )
    df.to_parquet(proc / "train.parquet", index=False)
    df.to_parquet(proc / "test.parquet", index=False)

    man = build_manifest(tmp_path / "processed", "mathdial", seed=7)
    assert len(man.train_ids) + len(man.dev_ids) + len(man.test_ids) == man.n_total
    path = write_manifest(man, tmp_path / "splits")
    assert path.exists()


def test_manifest_fold_ids() -> None:
    m = SplitManifest(
        dataset="x",
        seed=1,
        train_ids=["a"],
        dev_ids=["b"],
        test_ids=["c", "d"],
        stratify="t",
        n_total=4,
    )
    assert manifest_fold_ids(m, "train") == frozenset({"a"})
    assert manifest_fold_ids(m, "DEV") == frozenset({"b"})
    assert len(manifest_fold_ids(m, "test")) == 2


def test_manifest_fold_ids_invalid() -> None:
    m = SplitManifest(
        dataset="x",
        seed=1,
        train_ids=[],
        dev_ids=[],
        test_ids=[],
        stratify="t",
        n_total=0,
    )
    with pytest.raises(ValueError, match="fold"):
        manifest_fold_ids(m, "validation")
