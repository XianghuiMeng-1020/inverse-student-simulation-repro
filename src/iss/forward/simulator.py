"""Greedy forward roll-out using a PEFT-tuned Qwen causal LM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from omegaconf import DictConfig
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from iss.forward.prompts import build_forward_messages
from iss.schema.latent import DialogueTurn, LatentZ


@dataclass
class ForwardSimulator:
    """Minimal batched interface (single sequence)."""

    model: Any
    tokenizer: Any
    gen_cfg: GenerationConfig

    @classmethod
    def from_checkpoint(
        cls,
        cfg: DictConfig,
        *,
        adapter_dir: str,
        max_gpu_gb: float | None = None,
    ) -> ForwardSimulator:
        torch_dtype = getattr(torch, str(cfg.model.dtype))

        load_kwargs: dict[str, Any] = {
            "attn_implementation": str(cfg.model.attn_implementation),
            "trust_remote_code": bool(cfg.model.trust_remote_code),
        }
        if max_gpu_gb is not None and max_gpu_gb > 0:
            # Cap GPU memory and quantize to 4-bit so we coexist with other processes.
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            load_kwargs["device_map"] = "auto"
            load_kwargs["max_memory"] = {
                0: f"{int(max_gpu_gb)}GiB",
                "cpu": "40GiB",
            }
        else:
            load_kwargs["torch_dtype"] = torch_dtype
            load_kwargs["device_map"] = "auto"

        base = AutoModelForCausalLM.from_pretrained(cfg.model.backbone, **load_kwargs)
        model = PeftModel.from_pretrained(base, adapter_dir)
        model.eval()
        tok = AutoTokenizer.from_pretrained(cfg.model.tokenizer, trust_remote_code=False)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        g = cfg.model.generation
        gen = GenerationConfig(
            max_new_tokens=int(g.max_new_tokens),
            temperature=float(g.temperature),
            top_p=float(g.top_p),
            do_sample=float(g.temperature) > 0.0,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
        return cls(model=model, tokenizer=tok, gen_cfg=gen)

    @torch.inference_mode()
    def next_student_line(
        self,
        *,
        z: LatentZ,
        prefix_turns: list[DialogueTurn],
    ) -> str:
        messages = build_forward_messages(z=z, prefix_turns=prefix_turns)
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(next(self.model.parameters()).device)
        out = self.model.generate(**inputs, generation_config=self.gen_cfg)
        gen_ids = out[0, inputs["input_ids"].shape[1] :]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        return text
