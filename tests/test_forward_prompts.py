"""Forward / pseudo-Z smoke tests (no HF download)."""

from __future__ import annotations

from iss.forward.prompts import build_forward_messages
from iss.forward.pseudo_z import pseudo_latent_z_from_prefix
from iss.schema.grammar import validate_latent_z_json
from iss.schema.latent import DialogueTurn


def test_pseudo_z_validates() -> None:
    prefix = [
        DialogueTurn(turn_index=0, speaker="tutor", text="What is 1/2 + 1/4?"),
    ]
    z = pseudo_latent_z_from_prefix(prefix)
    obj = z.model_dump(mode="json", exclude={"rationale"})
    z2 = validate_latent_z_json(obj)
    assert z2.mastery.values["KC07"] > 0.5


def test_forward_messages_roles() -> None:
    prefix = [
        DialogueTurn(turn_index=0, speaker="tutor", text="Compute 3+5."),
    ]
    z = pseudo_latent_z_from_prefix(prefix)
    msgs = build_forward_messages(z=z, prefix_turns=prefix)
    assert msgs[0]["role"] == "system"
    assert "Latent state JSON" in msgs[1]["content"]
