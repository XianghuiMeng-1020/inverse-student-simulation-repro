#!/usr/bin/env python3
"""Verify SHA-256 checksums for downloaded raw data (plan p19-08).

Manifest JSON format::

    {
      "root": ".",
      "files": [
        {"path": "data/raw/mathdial/README.md", "sha256": "<hex>"},
        ...
      ]
    }

Paths are relative to ``root`` (default: repository root inferred from this file).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def sha256_file(path: Path, *, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_manifest(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "files" not in data or not isinstance(data["files"], list):
        msg = "manifest must contain a 'files' list"
        raise ValueError(msg)
    return data


def verify(manifest_path: Path, *, strict: bool) -> int:
    data = load_manifest(manifest_path)
    root = Path(data.get("root", "."))
    if not root.is_absolute():
        root = (_repo_root() / root).resolve()
    failures: list[str] = []
    missing: list[str] = []
    for entry in data["files"]:
        rel = entry["path"]
        want = entry["sha256"].lower()
        fp = (root / rel).resolve()
        if not fp.is_file():
            missing.append(str(fp))
            continue
        got = sha256_file(fp).lower()
        if got != want:
            failures.append(f"{rel}: expected {want}, got {got}")
    if missing:
        for m in missing:
            print(f"MISSING: {m}", file=sys.stderr)
    if failures:
        for f in failures:
            print(f"MISMATCH: {f}", file=sys.stderr)
    if not missing and not failures:
        print(f"OK: {len(data['files'])} files match manifest {manifest_path}")
        return 0
    if strict:
        return 2
    return 1


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "manifest",
        type=Path,
        nargs="?",
        default=_repo_root() / "data" / "checksums.json",
        help="Path to checksum manifest JSON",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Exit 2 if any file missing or mismatch (default: 1 on any issue)",
    )
    args = p.parse_args()
    if not args.manifest.is_file():
        print(f"No manifest at {args.manifest}; create data/checksums.json first.", file=sys.stderr)
        sys.exit(3)
    sys.exit(verify(args.manifest, strict=args.strict))


if __name__ == "__main__":
    main()
