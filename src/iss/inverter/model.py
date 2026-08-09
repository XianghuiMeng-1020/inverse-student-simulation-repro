"""Joint inverter model wrapper (plan p6-01 scaffold; SFT weights optional)."""

from __future__ import annotations

import json
import logging
import re

import torch
from omegaconf import DictConfig, OmegaConf
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
)

from iss.inverter.prompts import build_inverter_messages
from iss.schema.grammar import validate_latent_z_json
from iss.schema.kc_ontology import get_kc_ids
from iss.schema.latent import DialogueTurn, LatentZ
from iss.schema.misconception_catalogue import get_misconception_ids

_log = logging.getLogger(__name__)


class JointInverter:
    """Causal LM that emits ``LatentZ`` JSON (teacher-forced during training)."""

    def __init__(
        self,
        cfg: DictConfig,
        *,
        adapter_dir: str | None = None,
        force_cpu: bool | None = None,
    ) -> None:
        torch_dtype = getattr(torch, str(cfg.model.dtype))
        on_cpu = bool(force_cpu if force_cpu is not None else getattr(cfg.model, "force_cpu", False))
        use_4bit = bool(getattr(cfg.model, "load_in_4bit", False)) and not on_cpu
        max_gpu_gb = OmegaConf.select(cfg, "model.max_gpu_gb", default=None)
        load_kwargs: dict = {
            "attn_implementation": str(cfg.model.attn_implementation),
            "trust_remote_code": bool(cfg.model.trust_remote_code),
            "low_cpu_mem_usage": True,
        }
        if on_cpu:
            load_kwargs["device_map"] = "cpu"
            load_kwargs["torch_dtype"] = torch.float16
        else:
            load_kwargs["device_map"] = "auto" if torch.cuda.is_available() else "cpu"
        if use_4bit:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif not on_cpu:
            load_kwargs["torch_dtype"] = torch_dtype
        if max_gpu_gb and torch.cuda.is_available() and not on_cpu:
            load_kwargs["max_memory"] = {0: f"{int(max_gpu_gb)}GiB", "cpu": "48GiB"}
        base = AutoModelForCausalLM.from_pretrained(cfg.model.backbone, **load_kwargs)
        if adapter_dir:
            self.model = PeftModel.from_pretrained(base, adapter_dir, is_trainable=False)
        else:
            self.model = base
        if on_cpu:
            self.model = self.model.to("cpu")
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model.tokenizer,
            trust_remote_code=False,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.gen = GenerationConfig(
            max_new_tokens=int(cfg.model.max_output_tokens),
            temperature=0.0,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        self._max_prompt_tokens = int(cfg.model.max_input_tokens)

    @staticmethod
    def _parse_z(text: str) -> LatentZ:
        s = text.strip()
        if s.startswith("```"):
            s = s.split("\n", 1)[-1]
            if "```" in s:
                s = s.rsplit("```", 1)[0]
        for _ in range(3):
            s = re.sub(r'\}\},?\s*\{"(misconceptions|metacog)"', r'}}, "\1"', s)
            s = re.sub(r'\},?\s*\{"(metacog)"', r'}, "\1"', s)

        default_metacog = {
            "monitoring_accuracy": 0.5,
            "help_seeking_ratio": 0.5,
            "confidence_correctness_gap": 0.0,
            "hint_uptake": 0.5,
        }

        def _fill(obj: dict) -> dict:
            if "mastery" not in obj:
                obj["mastery"] = {"values": {k: 0.5 for k in get_kc_ids()}}
            if "misconceptions" not in obj:
                obj["misconceptions"] = {"probs": {m: 0.0 for m in get_misconception_ids()}}
            if "metacog" not in obj:
                obj["metacog"] = default_metacog
            valid_kc = set(get_kc_ids())
            if isinstance(obj.get("mastery"), dict) and "values" in obj["mastery"]:
                obj["mastery"]["values"] = {
                    k: v for k, v in obj["mastery"]["values"].items() if k in valid_kc
                }
            return obj

        for parser in (
            lambda t: json.loads(t),
            lambda t: json.JSONDecoder().raw_decode(t)[0],
        ):
            try:
                return validate_latent_z_json(_fill(parser(s)))
            except Exception:
                continue
        _log.warning("Failed to parse Z JSON: %s", s[:200])
        return validate_latent_z_json(
            _fill(
                {
                    "mastery": {"values": {k: 0.5 for k in get_kc_ids()}},
                    "misconceptions": {"probs": {m: 0.0 for m in get_misconception_ids()}},
                    "metacog": default_metacog,
                }
            )
        )

    @torch.inference_mode()
    def generate_z(self, turns: list[DialogueTurn]) -> LatentZ:
        messages = build_inverter_messages(dialogue_turns=turns)
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        dev = next(self.model.parameters()).device
        enc = self.tokenizer(prompt, return_tensors="pt").to(dev)
        out = self.model.generate(**enc, generation_config=self.gen)
        gen_ids = out[0, enc["input_ids"].shape[1] :]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        return self._parse_z(text)

    @torch.inference_mode()
    def generate_z_batch(self, turns_list: list[list[DialogueTurn]]) -> list[LatentZ]:
        """Generate LatentZ for multiple turn-prefixes in one GPU batch.

        Uses left-padding so all sequences are right-aligned for causal generation.
        Much faster than sequential generate_z when the GPU has available memory.
        """
        if not turns_list:
            return []

        original_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"

        prompts = [
            self.tokenizer.apply_chat_template(
                build_inverter_messages(dialogue_turns=turns),
                tokenize=False,
                add_generation_prompt=True,
            )
            for turns in turns_list
        ]

        dev = next(self.model.parameters()).device
        enc = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self._max_prompt_tokens,
        ).to(dev)

        out = self.model.generate(**enc, generation_config=self.gen)

        self.tokenizer.padding_side = original_padding_side

        results: list[LatentZ] = []
        prompt_len = enc["input_ids"].shape[1]
        for i in range(len(turns_list)):
            gen_ids = out[i, prompt_len:]
            text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            results.append(self._parse_z(text))
        return results
