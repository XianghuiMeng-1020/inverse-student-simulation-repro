"""Tutorial step 2: build Pseudo-Z and Random-Z from toy dialogues.

No GPU, no API key required. Run:

    python tutorial/02_construct_states.py

This mirrors the same functions used at full scale in the paper
(``iss.forward.pseudo_z``) on the three synthetic toy dialogues in
``tutorial/toy_dialogues.json`` and writes the resulting states to
``tutorial/outputs/states.json`` for inspection.

Pseudo-Z is a deterministic, evidence-free heuristic used as (a) an
early-bootstrap weak label and (b) a label-prior baseline in the paper's
misconception sparsity audit. Random-Z is the E4 counterfactual control
arm (IID Uniform[0,1] per field), used to test whether a forward model's
next-turn predictions are sensitive to Z at all.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from iss.forward.pseudo_z import pseudo_latent_z_from_prefix, uniform_random_latent_z  # noqa: E402
from iss.schema.latent import DialogueTurn  # noqa: E402


def load_toy_dialogues() -> list[dict]:
    path = Path(__file__).with_name("toy_dialogues.json")
    return json.loads(path.read_text(encoding="utf-8"))


def to_turns(raw_turns: list[dict]) -> list[DialogueTurn]:
    return [
        DialogueTurn(turn_index=i, speaker=t["speaker"], text=t["text"])
        for i, t in enumerate(raw_turns)
    ]


def main() -> None:
    dialogues = load_toy_dialogues()
    out_dir = Path(__file__).with_name("outputs")
    out_dir.mkdir(exist_ok=True)

    records = []
    for i, dlg in enumerate(dialogues):
        turns = to_turns(dlg["turns"])
        pseudo_z = pseudo_latent_z_from_prefix(turns)
        random_z = uniform_random_latent_z(seed=1000 + i)

        n_nonflat_kc = sum(1 for v in pseudo_z.mastery.values.values() if abs(v - 0.5) > 1e-9)
        print(f"[{dlg['dialogue_id']}] pseudo-Z: {n_nonflat_kc}/30 KCs shifted from the 0.5 prior "
              f"by lexical cues; help_seeking_ratio={pseudo_z.metacog.help_seeking_ratio:.2f}")

        records.append(
            {
                "dialogue_id": dlg["dialogue_id"],
                "pseudo_z": pseudo_z.model_dump(mode="json"),
                "random_z": random_z.model_dump(mode="json"),
            }
        )

    out_path = out_dir / "states.json"
    out_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"\n[done] wrote {len(records)} Pseudo-Z/Random-Z pairs to {out_path}")


if __name__ == "__main__":
    main()
