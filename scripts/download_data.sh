#!/usr/bin/env bash
# One-command-ish raw data fetch (plan p19-08). Extend per-dataset as URLs stabilize.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
echo "ISS download_data: populate data/raw/* via dataset-specific instructions in data/DATASHEET.md."
echo "After downloading, update data/checksums.json and run: uv run python scripts/verify_checksums.py"
exit 0
