"""Tutorial step 1: inspect the structured student-state schema Z = (m, C, g).

No GPU, no API key, no downloaded data required. Run:

    python tutorial/01_inspect_schema.py

This prints the fixed ontology sizes used throughout the paper (30 KCs,
70 misconception codes, 4 metacognitive scalars) and validates that the
Pydantic schema + JSON-repair path round-trips correctly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from iss.schema.grammar import validate_latent_z_json  # noqa: E402
from iss.schema.kc_ontology import get_kc_ids, load_knowledge_components  # noqa: E402
from iss.schema.misconception_catalogue import get_misconception_ids  # noqa: E402
from iss.schema.repair import repair_latent_z_json  # noqa: E402


def main() -> None:
    kc_ids = get_kc_ids()
    misc_ids = get_misconception_ids()

    print(f"[ontology] {len(kc_ids)} knowledge components (expect 30): {kc_ids[:5]} ...")
    print(f"[ontology] {len(misc_ids)} misconception codes (expect 70): {misc_ids[:5]} ...")

    kcs = load_knowledge_components()
    print(f"\n[example KC] {kcs[0].id} -- {kcs[0].name}\n  {kcs[0].description}")

    # A deliberately non-canonical model output -- aliased/short key names
    # (e.g. "KC1" instead of "KC01", "M7" instead of "M007") and missing
    # entries, exactly the kind of drift a fine-tuned LLM produces in
    # practice -- to exercise deterministic repair before schema validation.
    raw_model_output = {
        "mastery": {f"KC{i}": 0.5 for i in range(1, 30)},  # short keys, KC30 missing
        "misconceptions": {"M1": 0.4, "M7": 0.1, "M23": 0.05},  # short keys, most missing
        "metacog": {
            "monitoring": 0.4,  # alias for monitoring_accuracy
            "help-seeking": 0.2,  # alias for help_seeking_ratio
            "conf_gap": 0.1,  # alias for confidence_correctness_gap
            "hint": 0.3,  # alias for hint_uptake
        },
        "unexpected_field": "should be dropped",
    }
    repaired = repair_latent_z_json(raw_model_output)
    z = validate_latent_z_json(repaired)
    print(f"\n[repair] recovered {len(z.mastery.values)} KC entries, "
          f"{len(z.misconceptions.probs)} misconception entries after repair.")
    print("[repair] sample repaired JSON (truncated):")
    print(json.dumps(z.model_dump(mode="json"), indent=2)[:400] + " ...")


if __name__ == "__main__":
    main()
