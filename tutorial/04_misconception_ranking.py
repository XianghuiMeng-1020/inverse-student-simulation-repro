"""Tutorial step 4: misconception ranking metrics and the label-prior baseline.

No GPU, no API key required. Run:

    python tutorial/02_construct_states.py   # (if not already run)
    python tutorial/04_misconception_ranking.py

The manuscript's central misconception-facet finding is a *sparsity
caveat*: because all 394 real MathDial test dialogues share an identical
4-code active misconception set in silver v3, a trivial label-prior
baseline (always predict the same fixed set, ignoring the dialogue)
scores almost as well as the fine-tuned inverter on F1@5 / MRR. This
script reproduces that comparison mechanically on the toy dialogues so
you can see *why* near-perfect ranking metrics do not, by themselves,
demonstrate genuine per-dialogue discrimination.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from iss.eval.metrics import f1_at_k, mrr_at_k  # noqa: E402
from iss.schema.latent import LatentZ  # noqa: E402
from iss.schema.misconception_catalogue import get_misconception_ids  # noqa: E402


def load_states() -> list[dict]:
    path = Path(__file__).with_name("outputs") / "states.json"
    if not path.is_file():
        raise SystemExit("Run tutorial/02_construct_states.py first.")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    records = load_states()
    misc_ids = get_misconception_ids()

    # Toy "gold active set": pretend every toy dialogue actually activates
    # the same 2 misconception codes (mirrors the real sparsity finding).
    gold_active = {"M001", "M002"}
    gold_binary = np.array([1 if m in gold_active else 0 for m in misc_ids])

    label_prior_score = gold_binary.astype(float)  # always predicts the fixed active set

    f1_prior, f1_model = [], []
    mrr_prior, mrr_model = [], []
    for rec in records:
        pred_z = LatentZ.model_validate(rec["pseudo_z"])
        pred_score = np.array([pred_z.misconceptions.probs[m] for m in misc_ids])

        f1_model.append(f1_at_k(gold_binary, pred_score, k=5))
        f1_prior.append(f1_at_k(gold_binary, label_prior_score, k=5))

        ranked_model = list(np.argsort(-pred_score))
        ranked_prior = list(np.argsort(-label_prior_score))
        positives = {i for i, g in enumerate(gold_binary) if g == 1}
        mrr_model.append(mrr_at_k(ranked_model, positives, k=10))
        mrr_prior.append(mrr_at_k(ranked_prior, positives, k=10))

    print("[toy misconception ranking, n=3 dialogues, gold active set = {M001, M002}]")
    print(f"  F1@5  pseudo-Z (dialogue-aware) = {np.mean(f1_model):.3f}")
    print(f"  F1@5  label-prior (fixed set)   = {np.mean(f1_prior):.3f}")
    print(f"  MRR@10 pseudo-Z                 = {np.mean(mrr_model):.3f}")
    print(f"  MRR@10 label-prior               = {np.mean(mrr_prior):.3f}")
    print(
        "\n[interpretation] if the label-prior baseline (which never looks at "
        "the dialogue) matches or beats the dialogue-aware prediction, high "
        "F1/MRR reflects label-set uniformity rather than genuine per-dialogue "
        "misconception discrimination -- exactly the caveat raised in the "
        "manuscript for the real 394-dialogue evaluation. See "
        "scripts/robustness_misconception_sparsity.py for the full audit."
    )


if __name__ == "__main__":
    main()
