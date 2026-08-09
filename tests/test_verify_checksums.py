"""Integration tests for ``scripts/verify_checksums.py``."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_verify_checksums_ok(tmp_path: Path) -> None:
    f = tmp_path / "hello.txt"
    f.write_bytes(b"hello\n")
    import hashlib

    h = hashlib.sha256(b"hello\n").hexdigest()
    man = tmp_path / "m.json"
    man.write_text(
        json.dumps({"root": str(tmp_path), "files": [{"path": "hello.txt", "sha256": h}]}),
        encoding="utf-8",
    )
    r = subprocess.run(
        [sys.executable, str(Path(__file__).resolve().parents[1] / "scripts" / "verify_checksums.py"), str(man), "--strict"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0
    assert "OK" in r.stdout


def test_verify_checksums_mismatch(tmp_path: Path) -> None:
    f = tmp_path / "a.txt"
    f.write_bytes(b"x")
    man = tmp_path / "m.json"
    man.write_text(
        json.dumps({"root": str(tmp_path), "files": [{"path": "a.txt", "sha256": "0" * 64}]}),
        encoding="utf-8",
    )
    r = subprocess.run(
        [sys.executable, str(Path(__file__).resolve().parents[1] / "scripts" / "verify_checksums.py"), str(man), "--strict"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 2
