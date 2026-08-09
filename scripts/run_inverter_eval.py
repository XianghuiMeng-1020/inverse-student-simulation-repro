"""E1: Inverter accuracy evaluation on MathDial test set.

Runs the silver-trained inverter at multiple dialogue prefix lengths, computes
KC mastery Brier/AUC, misconception F1@k/MRR, and metacog Pearson r against
silver labels. Also computes pseudo-Z heuristic as a baseline.

Usage:
    python scripts/run_inverter_eval.py \
        --adapter-dir experiments/checkpoints/inverter_3b_qlora_silver_v1/inverter_lora \
        --silver-labels data/labels/latent_z_silver_core.jsonl \
        --max-gpu-gb 8
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

from iss.data.latent_labels import load_latent_z_label_jsonl
from iss.forward.pseudo_z import pseudo_latent_z_from_prefix
from iss.schema.kc_ontology import get_kc_ids
from iss.schema.latent import DialogueTurn, LatentZ
from iss.schema.misconception_catalogue import get_misconception_ids

log = logging.getLogger("e1_eval")

# Hard-coded from configs/model/inverter_3b.yaml
BACKBONE = "Qwen/Qwen2.5-3B-Instruct"
TOKENIZER = "Qwen/Qwen2.5-3B-Instruct"
MAX_INPUT_TOKENS = 1024
MAX_OUTPUT_TOKENS = 1280  # KC×30 + Misc×70 + metacog ≈ 720 tokens; 1280 for safe headroom
ATTN_IMPL = "sdpa"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_test_dialogues(parquet_path: Path) -> list[dict[str, Any]]:
    df = pd.read_parquet(parquet_path)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for _, row in df.iterrows():
        did = str(row["dialogue_id"])
        if did in seen:
            continue
        seen.add(did)
        turns_raw: list[dict[str, Any]] = json.loads(row["turns_json"])
        turns = [DialogueTurn(**t) for t in turns_raw]
        rows.append({"dialogue_id": did, "turns": turns})
    return rows


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    pred_zs: list[LatentZ],
    gold_zs: list[LatentZ],
    label: str = "full",
) -> dict[str, Any]:
    kc_ids = get_kc_ids()
    misc_ids = get_misconception_ids()
    metacog_fields = [
        "monitoring_accuracy",
        "help_seeking_ratio",
        "confidence_correctness_gap",
        "hint_uptake",
    ]

    # --- KC mastery Brier and AUC ---
    kc_briers: list[float] = []
    kc_aucs: list[float] = []
    for kc in kc_ids:
        pred_vals = np.array([z.mastery.values.get(kc, 0.5) for z in pred_zs])
        gold_vals = np.array([z.mastery.values.get(kc, 0.5) for z in gold_zs])
        kc_briers.append(float(np.mean((pred_vals - gold_vals) ** 2)))
        gold_bin = (gold_vals > 0.5).astype(int)
        if 0 < gold_bin.sum() < len(gold_bin):
            try:
                kc_aucs.append(float(roc_auc_score(gold_bin, pred_vals)))
            except Exception:
                pass

    # --- Misconception F1@k and MRR@10 ---
    f1_5_list: list[float] = []
    f1_10_list: list[float] = []
    mrr_list: list[float] = []
    for pred_z, gold_z in zip(pred_zs, gold_zs, strict=False):
        pred_sorted = sorted(
            misc_ids, key=lambda m: pred_z.misconceptions.probs.get(m, 0.0), reverse=True
        )
        gold_sorted = sorted(
            misc_ids, key=lambda m: gold_z.misconceptions.probs.get(m, 0.0), reverse=True
        )
        gold_top5 = set(gold_sorted[:5])
        gold_top10 = set(gold_sorted[:10])
        pred_top5 = set(pred_sorted[:5])
        pred_top10 = set(pred_sorted[:10])

        tp5 = len(pred_top5 & gold_top5)
        p5, r5 = tp5 / 5, tp5 / max(len(gold_top5), 1)
        f1_5_list.append(2 * p5 * r5 / (p5 + r5) if (p5 + r5) > 0 else 0.0)

        tp10 = len(pred_top10 & gold_top10)
        p10, r10 = tp10 / 10, tp10 / max(len(gold_top10), 1)
        f1_10_list.append(2 * p10 * r10 / (p10 + r10) if (p10 + r10) > 0 else 0.0)

        mrr = 0.0
        for rank, m in enumerate(pred_sorted[:10], 1):
            if m in gold_top10:
                mrr = 1.0 / rank
                break
        mrr_list.append(mrr)

    # --- Metacog Pearson r ---
    metacog_r: dict[str, float] = {}
    for field in metacog_fields:
        pv = np.array([getattr(z.metacog, field) for z in pred_zs])
        gv = np.array([getattr(z.metacog, field) for z in gold_zs])
        if np.std(pv) > 1e-8 and np.std(gv) > 1e-8:
            r_val, _ = pearsonr(pv, gv)
            metacog_r[field] = float(r_val)
        else:
            metacog_r[field] = float("nan")

    return {
        "prefix": label,
        "n": len(pred_zs),
        "kc_brier_mean": float(np.mean(kc_briers)),
        "kc_auc_mean": float(np.mean(kc_aucs)) if kc_aucs else float("nan"),
        "misc_f1_at_5": float(np.mean(f1_5_list)),
        "misc_f1_at_10": float(np.mean(f1_10_list)),
        "misc_mrr_at_10": float(np.mean(mrr_list)),
        "metacog_r": metacog_r,
        "metacog_r_mean": float(np.nanmean(list(metacog_r.values()))),
    }


# ---------------------------------------------------------------------------
# Inverter inference
# ---------------------------------------------------------------------------

def build_predictor(adapter_dir: Path, max_gpu_gb: float, batch_size: int = 8):
    """Return a callable: list[list[DialogueTurn]] -> list[LatentZ | None].

    When batch_size > 1 the function accepts a list of turn-lists and returns
    a list of predictions (same length). Falls back to single-item if needed.
    """
    import torch
    from peft import PeftModel
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        GenerationConfig,
    )

    from iss.inverter.prompts import build_inverter_messages
    from iss.schema.grammar import validate_latent_z_json

    # FP16 with per-sample empty_cache: ~40s/it and stable long-run memory.
    # KV-cache fragmentation is prevented by del enc + empty_cache in the predict finally block.
    log.info("Loading base model %s (FP16, max_gpu=%dGiB)...", BACKBONE, int(max_gpu_gb))
    base = AutoModelForCausalLM.from_pretrained(
        BACKBONE,
        dtype=torch.float16,
        device_map="auto",
        max_memory={0: f"{int(max_gpu_gb)}GiB", "cpu": "48GiB"},
        trust_remote_code=False,
        attn_implementation=ATTN_IMPL,
    )
    adapter_resolved = str(Path(adapter_dir).resolve())
    log.info("Loading LoRA adapter from %s...", adapter_resolved)
    model = PeftModel.from_pretrained(base, adapter_resolved)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    gen_config = GenerationConfig(
        max_new_tokens=MAX_OUTPUT_TOKENS,
        temperature=0.0,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    import re as _re

    _DEFAULT_METACOG = {
        "monitoring_accuracy": 0.5, "help_seeking_ratio": 0.5,
        "confidence_correctness_gap": 0.0, "hint_uptake": 0.5,
    }

    def _safe_validate(obj: dict) -> LatentZ | None:
        if "mastery" not in obj:
            obj["mastery"] = {"values": {k: 0.5 for k in get_kc_ids()}}
        if "misconceptions" not in obj:
            obj["misconceptions"] = {"probs": {m: 0.0 for m in get_misconception_ids()}}
        if "metacog" not in obj:
            obj["metacog"] = _DEFAULT_METACOG
        valid_kc = set(get_kc_ids())
        if isinstance(obj.get("mastery"), dict) and "values" in obj["mastery"]:
            obj["mastery"]["values"] = {
                k: v for k, v in obj["mastery"]["values"].items() if k in valid_kc
            }
        return validate_latent_z_json(obj)

    def _parse_output(text: str) -> LatentZ | None:
        for _ in range(3):
            text = _re.sub(r'\}\},?\s*\{"(misconceptions|metacog)"', r'}}, "\1"', text)
            text = _re.sub(r'\},?\s*\{"(metacog)"', r'}, "\1"', text)
        s = text.strip()
        if s.startswith("```"):
            s = s.split("\n", 1)[-1]
            if "```" in s:
                s = s.rsplit("```", 1)[0]
        try:
            return _safe_validate(json.loads(s))
        except Exception:
            pass
        try:
            obj, _ = json.JSONDecoder().raw_decode(s)
            return _safe_validate(obj)
        except Exception:
            pass
        try:
            merged: dict[str, object] = {}
            m_mastery = _re.search(r'"mastery"\s*:\s*\{\s*"values"\s*:\s*(\{[^}]+\})', s)
            if m_mastery:
                merged["mastery"] = {"values": json.loads(m_mastery.group(1))}
            m_misc = _re.search(r'"misconceptions"\s*:\s*\{\s*"probs"\s*:\s*(\{[^}]+\})', s)
            if m_misc:
                merged["misconceptions"] = {"probs": json.loads(m_misc.group(1))}
            m_meta = _re.search(r'"metacog"\s*:\s*(\{[^}]+\})', s)
            if m_meta:
                merged["metacog"] = json.loads(m_meta.group(1))
            if merged:
                return _safe_validate(merged)
        except Exception:
            pass
        log.warning("All parse strategies failed: %s", s[:200])
        return None

    def predict_batch(turns_list: list[list[DialogueTurn]]) -> list[LatentZ | None]:
        """Batched inference: process multiple dialogues in one GPU call."""
        import gc
        if not turns_list:
            return []

        original_padding_side = tokenizer.padding_side
        tokenizer.padding_side = "left"
        prompts = [
            tokenizer.apply_chat_template(
                build_inverter_messages(dialogue_turns=turns),
                tokenize=False,
                add_generation_prompt=True,
            )
            for turns in turns_list
        ]
        enc = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_INPUT_TOKENS,
        )
        tokenizer.padding_side = original_padding_side

        dev = next(model.parameters()).device
        enc = {k: v.to(dev) for k, v in enc.items()}
        prompt_len = enc["input_ids"].shape[1]

        try:
            with torch.inference_mode():
                out = model.generate(**enc, generation_config=gen_config)
        except torch.cuda.OutOfMemoryError:
            log.warning("CUDA OOM in batch (size=%d) — falling back one-by-one", len(turns_list))
            del enc
            torch.cuda.empty_cache()
            gc.collect()
            results = []
            for t in turns_list:
                results.append(_predict_single(t))
            return results
        finally:
            del enc
            torch.cuda.empty_cache()
            gc.collect()

        results = [_parse_output(tokenizer.decode(out[i, prompt_len:], skip_special_tokens=True))
                   for i in range(len(turns_list))]
        del out
        return results

    def _predict_single(turns: list[DialogueTurn]) -> LatentZ | None:
        """Single-sample fallback used by OOM recovery path."""
        import gc
        messages = build_inverter_messages(dialogue_turns=turns)
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = tokenizer(prompt, return_tensors="pt")
        if enc["input_ids"].shape[1] > MAX_INPUT_TOKENS:
            enc = {k: v[:, -MAX_INPUT_TOKENS:] for k, v in enc.items()}
        dev = next(model.parameters()).device
        enc = {k: v.to(dev) for k, v in enc.items()}
        input_len = enc["input_ids"].shape[1]
        try:
            with torch.inference_mode():
                out = model.generate(**enc, generation_config=gen_config)
            gen_ids = out[0, input_len:]
            text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            del out
            return _parse_output(text)
        except torch.cuda.OutOfMemoryError:
            log.warning("CUDA OOM — skipping sample")
            return None
        finally:
            del enc
            torch.cuda.empty_cache()
            gc.collect()

    def predict(turns: list[DialogueTurn]) -> LatentZ | None:
        return predict_batch([turns])[0]

    predict.batch = predict_batch  # type: ignore[attr-defined]
    return predict


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="E1 inverter accuracy evaluation")
    parser.add_argument(
        "--adapter-dir",
        default="experiments/checkpoints/inverter_3b_qlora_silver_v1/inverter_lora",
    )
    parser.add_argument("--silver-labels", default="data/labels/latent_z_silver_core.jsonl")
    parser.add_argument("--test-parquet", default="data/processed/mathdial/test.parquet")
    parser.add_argument("--out", default="experiments/results/e1_inverter_eval.json")
    parser.add_argument(
        "--prefix-fracs",
        type=float,
        nargs="+",
        default=None,
        help="(Legacy) Fraction of dialogue turns — prefer --prefix-turns.",
    )
    parser.add_argument(
        "--prefix-turns",
        type=int,
        nargs="+",
        default=[2, 4, 8, 0],
        help="Absolute prefix lengths in turns (0 = full dialogue).",
    )
    parser.add_argument("--max-gpu-gb", type=float, default=8.0)
    parser.add_argument("--batch-size", type=int, default=8, help="Inference batch size (samples per GPU call)")
    parser.add_argument("--limit", type=int, default=0, help="0 = all test dialogues")
    parser.add_argument("--skip-inverter", action="store_true", help="Only run pseudo-Z baseline")
    parser.add_argument(
        "--include-gpt4o",
        action="store_true",
        help="Add GPT-4o zero-shot inverter baseline (Tfull, needs OPENAI_API_KEY).",
    )
    parser.add_argument(
        "--gpt4o-model",
        type=str,
        default="gpt-4o",
        help="OpenAI chat model for --include-gpt4o.",
    )
    parser.add_argument(
        "--gpt4o-workers",
        type=int,
        default=16,
        help="Parallel API workers for --include-gpt4o (0 = sequential).",
    )
    parser.add_argument(
        "--include-bkt",
        action="store_true",
        help="Merge BKT baseline metrics from experiments/results/e1_bkt_baseline.json.",
    )
    args = parser.parse_args()

    from dotenv import load_dotenv

    load_dotenv(repo_root / ".env")

    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    # Load silver labels (gold for E1)
    silver_path = repo_root / args.silver_labels
    gold_map = load_latent_z_label_jsonl(silver_path)
    log.info("Loaded %d silver labels", len(gold_map))

    # Load test dialogues
    test_path = repo_root / args.test_parquet
    dialogues = load_test_dialogues(test_path)
    if args.limit > 0:
        dialogues = dialogues[: args.limit]
    dialogues = [d for d in dialogues if d["dialogue_id"] in gold_map]
    log.info("Test dialogues with gold labels: %d", len(dialogues))

    results_by_prefix: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Pseudo-Z baseline (no model needed)
    # ------------------------------------------------------------------
    log.info("Computing pseudo-Z heuristic baseline...")
    pseudo_preds: list[LatentZ] = []
    pseudo_golds: list[LatentZ] = []
    for d in tqdm(dialogues, desc="pseudo-Z"):
        pseudo_preds.append(pseudo_latent_z_from_prefix(d["turns"]))
        pseudo_golds.append(gold_map[d["dialogue_id"]])
    results_by_prefix["pseudo"] = compute_metrics(pseudo_preds, pseudo_golds, label="pseudo")
    m = results_by_prefix["pseudo"]
    log.info(
        "Pseudo: kc_brier=%.4f kc_auc=%.4f misc_f1@5=%.4f misc_mrr@10=%.4f metacog_r=%.4f",
        m["kc_brier_mean"], m["kc_auc_mean"], m["misc_f1_at_5"], m["misc_mrr_at_10"], m["metacog_r_mean"],
    )

    if args.include_bkt:
        bkt_path = repo_root / "experiments" / "results" / "e1_bkt_baseline.json"
        if bkt_path.is_file():
            bkt = json.loads(bkt_path.read_text(encoding="utf-8"))
            results_by_prefix["bkt"] = {
                "n": bkt.get("n", len(dialogues)),
                "kc_brier_mean": bkt.get("kc_brier_mean", float("nan")),
                "kc_auc_mean": bkt.get("kc_auc_mean", float("nan")),
                "misc_f1_at_5": float("nan"),
                "misc_mrr_at_10": float("nan"),
                "metacog_r_mean": float("nan"),
            }
            log.info("Merged BKT baseline from %s", bkt_path)
        else:
            log.warning("BKT baseline missing: %s (run scripts/run_bkt_baseline.py)", bkt_path)

    if args.include_gpt4o:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        from iss.baselines.gpt4o_zs import invert_dialogue_gpt4o
        from iss.schema.grammar import validate_latent_z_json

        gpt_dialogues = dialogues[: args.limit] if args.limit > 0 else dialogues
        workers = max(0, int(args.gpt4o_workers))
        log.info(
            "GPT zero-shot inverter (Tfull) model=%s workers=%d n=%d",
            args.gpt4o_model,
            workers,
            len(gpt_dialogues),
        )
        gpt_preds: list[LatentZ] = []
        gpt_golds: list[LatentZ] = []

        def _gpt_one(d: dict) -> tuple[str, LatentZ | None]:
            out = invert_dialogue_gpt4o(d["turns"], model=args.gpt4o_model)
            if out.get("status") != "ok":
                return d["dialogue_id"], None
            try:
                return d["dialogue_id"], validate_latent_z_json(out["latent"])
            except Exception:
                return d["dialogue_id"], None

        if workers <= 1:
            it = tqdm(gpt_dialogues, desc="gpt4o-zs")
            for d in it:
                _, z = _gpt_one(d)
                if z is not None:
                    gpt_preds.append(z)
                    gpt_golds.append(gold_map[d["dialogue_id"]])
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(_gpt_one, d) for d in gpt_dialogues]
                for fut in tqdm(as_completed(futures), total=len(futures), desc="gpt4o-zs"):
                    did, z = fut.result()
                    if z is not None:
                        gpt_preds.append(z)
                        gpt_golds.append(gold_map[did])
        if gpt_preds:
            results_by_prefix["gpt4o_zs"] = compute_metrics(gpt_preds, gpt_golds, label="gpt4o_zs")
        else:
            log.warning("GPT baseline produced no valid predictions (API key?)")

    if not args.skip_inverter:
        adapter_path = repo_root / args.adapter_dir
        predict = build_predictor(adapter_path, args.max_gpu_gb, batch_size=args.batch_size)
        predict_batch = predict.batch  # type: ignore[attr-defined]

        # Checkpoint path for incremental saves (same dir as --out, .ckpt.jsonl extension)
        out_path = repo_root / args.out
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ckpt_path = out_path.with_suffix(".ckpt.jsonl")

        # Load already-completed predictions from checkpoint
        ckpt_data: dict[str, list[dict]] = {}  # prefix_label -> list of {"did", "z_hat", "z_gold"}
        if ckpt_path.exists():
            for line in ckpt_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                lbl = row["prefix"]
                ckpt_data.setdefault(lbl, []).append(row)
            log.info("Resumed checkpoint: %s", {k: len(v) for k, v in ckpt_data.items()})

        ckpt_fh = ckpt_path.open("a", encoding="utf-8")

        if args.prefix_fracs:
            turn_specs = [(f"T{int(f * 100)}", f) for f in args.prefix_fracs]
        else:
            turn_specs = [
                (f"T{t}" if t > 0 else "Tfull", t) for t in args.prefix_turns
            ]

        for prefix_label, t_spec in turn_specs:
            use_frac = isinstance(t_spec, float)
            log.info(
                "Inverter inference prefix=%s (%s)...",
                prefix_label,
                "frac" if use_frac else "turns",
            )

            # Collect already-done dialogue IDs for this prefix
            done_ids: set[str] = {r["did"] for r in ckpt_data.get(prefix_label, [])}
            preds: list[LatentZ] = []
            golds: list[LatentZ] = []
            # Replay completed predictions first
            for r in ckpt_data.get(prefix_label, []):
                try:
                    from iss.schema.grammar import validate_latent_z_json
                    preds.append(validate_latent_z_json(r["z_hat"]))
                    golds.append(validate_latent_z_json(r["z_gold"]))
                except Exception:
                    pass

            n_fail = 0
            remaining = [d for d in dialogues if d["dialogue_id"] not in done_ids]
            log.info("%s: %d already done, %d remaining", prefix_label, len(done_ids), len(remaining))

            # Batch inference: process args.batch_size dialogues per GPU call
            bs = args.batch_size
            with tqdm(total=len(dialogues), initial=len(done_ids), desc=prefix_label) as pbar:
                for batch_start in range(0, len(remaining), bs):
                    batch = remaining[batch_start : batch_start + bs]
                    turns_batch = []
                    for d in batch:
                        turns = d["turns"]
                        if use_frac:
                            n_use = max(2, round(len(turns) * float(t_spec)))
                        else:
                            n_use = len(turns) if int(t_spec) <= 0 else min(len(turns), max(2, int(t_spec)))
                        turns_batch.append(turns[:n_use])

                    z_hats = predict_batch(turns_batch)
                    for d, z_hat in zip(batch, z_hats):
                        if z_hat is None:
                            n_fail += 1
                            ckpt_fh.write(json.dumps({
                                "prefix": prefix_label, "did": d["dialogue_id"], "fail": True,
                                "z_hat": None, "z_gold": None,
                            }) + "\n")
                        else:
                            preds.append(z_hat)
                            golds.append(gold_map[d["dialogue_id"]])
                            ckpt_fh.write(json.dumps({
                                "prefix": prefix_label,
                                "did": d["dialogue_id"],
                                "fail": False,
                                "z_hat": z_hat.model_dump(),
                                "z_gold": gold_map[d["dialogue_id"]].model_dump(),
                            }) + "\n")
                    ckpt_fh.flush()
                    pbar.update(len(batch))

            log.info("%s: success=%d fail=%d", prefix_label, len(preds), n_fail)
            metrics = compute_metrics(preds, golds, label=prefix_label)
            metrics["n_fail"] = n_fail
            results_by_prefix[prefix_label] = metrics
            m = metrics
            log.info(
                "%s: kc_brier=%.4f kc_auc=%.4f misc_f1@5=%.4f misc_mrr@10=%.4f metacog_r=%.4f",
                prefix_label, m["kc_brier_mean"], m["kc_auc_mean"],
                m["misc_f1_at_5"], m["misc_mrr_at_10"], m["metacog_r_mean"],
            )

        ckpt_fh.close()

    # Save final results
    out_path = repo_root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"results": results_by_prefix}, indent=2), encoding="utf-8")
    log.info("Saved to %s", out_path)

    # Print summary table
    print("\n=== E1 Inverter Accuracy Summary ===")
    print(f"{'Condition':<12} {'KC_Brier':>9} {'KC_AUC':>8} {'Misc_F1@5':>10} {'MRR@10':>8} {'Meta_r':>8}")
    print("-" * 60)
    for key, m in results_by_prefix.items():
        print(
            f"{key:<12} {m['kc_brier_mean']:>9.4f} {m['kc_auc_mean']:>8.4f} "
            f"{m['misc_f1_at_5']:>10.4f} {m['misc_mrr_at_10']:>8.4f} {m['metacog_r_mean']:>8.4f}"
        )


if __name__ == "__main__":
    main()
