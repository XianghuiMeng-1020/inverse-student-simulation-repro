"""Causal LM SFT utilities (no ``trl`` import — avoids Windows GBK issues in TRL)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from omegaconf import OmegaConf
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)


@dataclass
class ChatSFTExample:
    input_ids: list[int]
    labels: list[int]


class ChatSFTDataset(Dataset):
    """Supervised fine-tuning on chat messages (last assistant turn is the target)."""

    def __init__(
        self,
        records: list[dict[str, Any]],
        tokenizer: PreTrainedTokenizerBase,
        *,
        max_length: int,
    ) -> None:
        self.examples: list[ChatSFTExample] = []
        for rec in records:
            messages = rec["messages"]
            if not messages or messages[-1]["role"] != "assistant":
                continue
            prompt_msgs = messages[:-1]
            answer = messages[-1]["content"]
            prompt_ids = tokenizer.apply_chat_template(
                prompt_msgs,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors=None,
            )
            if not isinstance(prompt_ids, list):
                prompt_ids = list(prompt_ids)
            resp_ids = tokenizer.encode(answer, add_special_tokens=False)
            input_ids = prompt_ids + resp_ids
            labels = [-100] * len(prompt_ids) + resp_ids
            if len(input_ids) > max_length:
                overflow = len(input_ids) - max_length
                if overflow >= len(prompt_ids):
                    continue
                prompt_ids = prompt_ids[overflow:]
                labels = [-100] * len(prompt_ids) + resp_ids
                input_ids = prompt_ids + resp_ids
            self.examples.append(ChatSFTExample(input_ids=input_ids, labels=labels))

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        ex = self.examples[idx]
        return {"input_ids": ex.input_ids, "labels": ex.labels}


def pad_batch(
    batch: list[dict[str, Any]],
    *,
    pad_token_id: int,
    max_length: int,
) -> dict[str, torch.Tensor]:
    max_len = min(max_length, max(len(x["input_ids"]) for x in batch))
    input_ids = []
    labels = []
    attention = []
    for x in batch:
        ids = x["input_ids"][:max_len]
        lab = x["labels"][:max_len]
        pad_n = max_len - len(ids)
        input_ids.append(ids + [pad_token_id] * pad_n)
        labels.append(lab + [-100] * pad_n)
        attention.append([1] * len(ids) + [0] * pad_n)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention, dtype=torch.long),
    }


class PadCollator:
    def __init__(self, *, pad_token_id: int, max_length: int) -> None:
        self.pad_token_id = pad_token_id
        self.max_length = max_length

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        return pad_batch(batch, pad_token_id=self.pad_token_id, max_length=self.max_length)


def attach_lora(model: Any, lora_cfg: Any) -> Any:
    target_modules = list(lora_cfg.target_modules)
    peft_cfg = LoraConfig(
        r=int(lora_cfg.r),
        lora_alpha=int(lora_cfg.alpha),
        lora_dropout=float(lora_cfg.dropout),
        target_modules=target_modules,
        bias=str(lora_cfg.bias),
        task_type=TaskType.CAUSAL_LM,
    )
    return get_peft_model(model, peft_cfg)


def train_causal_lm_sft(
    *,
    cfg: Any,
    records: list[dict[str, Any]],
    output_dir: str,
    max_length: int,
) -> None:
    torch_dtype = getattr(torch, str(cfg.model.dtype))
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.tokenizer,
        trust_remote_code=bool(cfg.model.trust_remote_code),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_4bit = bool(getattr(cfg.model, "load_in_4bit", False))
    max_gpu_gb = OmegaConf.select(cfg, "train.max_gpu_gb", default=None)
    device_map: str | dict[str, int] | None = "auto" if torch.cuda.is_available() else None
    bnb_cfg = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        if use_4bit
        else None
    )
    load_kwargs: dict[str, Any] = {
        "attn_implementation": str(cfg.model.attn_implementation),
        "trust_remote_code": bool(cfg.model.trust_remote_code),
        "device_map": device_map,
        "quantization_config": bnb_cfg,
    }
    if not use_4bit:
        load_kwargs["dtype"] = torch_dtype
    if max_gpu_gb and torch.cuda.is_available():
        load_kwargs["max_memory"] = {0: f"{int(max_gpu_gb)}GiB", "cpu": "48GiB"}
        load_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(cfg.model.backbone, **load_kwargs)
    if use_4bit:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=bool(cfg.train.gradient_checkpointing)
        )
    elif bool(cfg.train.gradient_checkpointing):
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model = attach_lora(model, cfg.model.lora)

    ds = ChatSFTDataset(records, tokenizer, max_length=max_length)
    collator = PadCollator(pad_token_id=int(tokenizer.pad_token_id), max_length=max_length)

    targs = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=float(cfg.train.epochs),
        per_device_train_batch_size=int(cfg.train.batch_size),
        gradient_accumulation_steps=int(cfg.train.grad_accum_steps),
        learning_rate=float(cfg.train.lr),
        weight_decay=float(cfg.train.weight_decay),
        max_grad_norm=float(cfg.train.max_grad_norm),
        warmup_ratio=float(cfg.train.warmup_ratio),
        logging_steps=int(cfg.train.log_every_n_steps),
        save_steps=int(cfg.train.save_every_n_steps),
        eval_steps=int(OmegaConf.select(cfg, "train.eval_every_n_steps", default=200)),
        bf16=str(cfg.train.precision) == "bf16",
        fp16=str(cfg.train.precision) == "fp16",
        report_to=[],
        remove_unused_columns=False,
        save_total_limit=2,
        dataloader_num_workers=int(OmegaConf.select(cfg, "train.dataloader_num_workers", default=2)),
        dataloader_pin_memory=True,
        optim="adamw_torch_fused",
        lr_scheduler_type=str(OmegaConf.select(cfg, "train.lr_scheduler", default="cosine")),
    )
    try:
        trainer = Trainer(
            model=model,
            args=targs,
            train_dataset=ds,
            data_collator=collator,
            processing_class=tokenizer,
        )
    except TypeError:
        trainer = Trainer(
            model=model,
            args=targs,
            train_dataset=ds,
            data_collator=collator,
            tokenizer=tokenizer,
        )
    resume_ckpt = OmegaConf.select(cfg, "train.resume_from_checkpoint", default=None)
    if resume_ckpt:
        trainer.train(resume_from_checkpoint=str(resume_ckpt))
    else:
        trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
