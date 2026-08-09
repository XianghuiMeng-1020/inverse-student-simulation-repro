"""Which corpora are in-scope for ICCE 2026 ISS experiments (all public)."""

from __future__ import annotations

from typing import Final

# Exactly four dialogue sources for cross-domain + metacog coverage.
CORE_DATASETS: Final[tuple[str, ...]] = ("mathdial", "bridge", "cima", "talkmoves")

DATASET_LICENSES: Final[dict[str, str]] = {
    "mathdial": "CC-BY-4.0 (Hugging Face eth-nlped/mathdial)",
    "bridge": "See rose-e-wang/bridge dataset card on Hugging Face",
    "cima": "CC BY-NC-SA 2.5 (kstats/CIMA GitHub README; verify before redistribution)",
    "talkmoves": "CC BY-NC-SA 4.0 (SumnerLab/TalkMoves LICENSE)",
}
