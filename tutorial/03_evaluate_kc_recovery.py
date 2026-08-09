"""Tutorial step 3: KC-mastery recovery metrics on toy data.

No GPU, no API key required. Run:

    python tutorial/02_construct_states.py   # (if not already run)
    python tutorial/03_evaluate_kc_recovery.py

This reproduces, at toy scale, the same metric definitions used for the
paper's E1 inversion-accuracy table and KC-structure analysis:
Brier score, ECE, within-dialogue Spearman correlation, and top-3
high-mastery KC overlap between a "prediction" and a reference state.

IMPORTANT: there are only 3 toy dialogues and no real annotations here,
so the numbers below are illustrative only -- they demonstrate the
*metric machinery*, not the paper's reported results. For the real
394-dialogue MathDial evaluation see scripts/run_inverter_eval.py and
scripts/robustness_kc_structure.py (README.md, "Reproducing Main
Results").
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from iss.eval.metrics import binary_brier, ece  # noqa: E402
from iss.forward.pseudo_z import uniform_random_latent_z  # noqa: E402
from iss.schema.kc_ontology import get_kc_ids  # noqa: E402
from iss.schema.latent import LatentZ  # noqa: E402


def load_states() -> list[dict]:
    path = Path(__file__).with_name("outputs") / "states.json"
    if not path.is_file():
        raise SystemExit("Run tutorial/02_construct_states.py first.")
    return json.loads(path.read_text(encoding="utf-8"))


def toy_reference_z(seed: int) -> LatentZ:
    """A stand-in 'reference' state for illustration only (not real silver labels)."""
    return uniform_random_latent_z(seed=seed)


def main() -> None:
    records = load_states()
    kc_ids = get_kc_ids()

    all_pred, all_gold_binary = [], []
    spearmans, top3_overlaps = [], []

    for i, rec in enumerate(records):
        pred_z = LatentZ.model_validate(rec["pseudo_z"])
        gold_z = toy_reference_z(seed=42 + i)  # illustrative reference, seeded for determinism

        pred_vals = np.array([pred_z.mastery.values[k] for k in kc_ids])
        gold_vals = np.array([gold_z.mastery.values[k] for k in kc_ids])
        gold_binary = (gold_vals > 0.5).astype(int)

        all_pred.extend(pred_vals.tolist())
        all_gold_binary.extend(gold_binary.tolist())

        rho, _ = spearmanr(pred_vals, gold_vals)
        spearmans.append(rho)

        top3_pred = set(np.argsort(-pred_vals)[:3].tolist())
        top3_gold = set(np.argsort(-gold_vals)[:3].tolist())
        overlap = len(top3_pred & top3_gold) / 3.0
        top3_overlaps.append(overlap)

        print(f"[{rec['dialogue_id']}] within-dialogue Spearman rho={rho:.3f}, "
              f"top-3 overlap={overlap:.2f}")

    brier = binary_brier(all_gold_binary, all_pred)
    calib = ece(all_pred, all_gold_binary, n_bins=5)

    print("\n[pooled, n=3 toy dialogues]")
    print(f"  KC Brier          = {brier:.4f}")
    print(f"  KC ECE (5 bins)   = {calib:.4f}")
    print(f"  mean Spearman rho = {np.nanmean(spearmans):.4f}")
    print(f"  mean top-3 overlap= {np.nanmean(top3_overlaps):.4f}")
    print(
        "\n[reminder] the reference state here is a seeded random draw for "
        "demonstration only; compare against results/expected_metrics.json "
        "for the actual paper numbers computed on 394 MathDial dialogues."
    )


if __name__ == "__main__":
    main()
