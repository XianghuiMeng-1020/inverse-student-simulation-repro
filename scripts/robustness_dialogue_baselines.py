"""
Task 3: Dialogue evidence baselines for inversion.

Using the existing inverter checkpoint jsonl, computes KC Brier, ECE, AUC, and
misconception MRR for lightweight ablation baselines derived from the inverter
predictions at different context lengths as proxies for:
  - Prior only: pseudo-Z baseline (no dialogue evidence)
  - Short context (T2): problem + first student turn only
  - Medium context (T4, T8): increasing dialogue evidence
  - Full dialogue (Tfull): complete transcript

Also compares these to BKT and gpt-4o zero-shot results.

The key question: does the inverter gain from student dialogue evidence beyond
problem/tutor priors?

Output: experiments/results/robustness_dialogue_baselines.json
        experiments/results/robustness_dialogue_baselines.md
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

repo_root = Path(__file__).resolve().parents[1]
RESULTS_DIR = repo_root / "experiments" / "results"
INV_V3_JSON = RESULTS_DIR / "e1_inverter_eval_v3.json"
EXTENDED_JSON = RESULTS_DIR / "e1_extended_v3.json"
GPT4O_JSON = RESULTS_DIR / "e1_gpt4o_zs_v3.json"
BKT_JSON = RESULTS_DIR / "e1_bkt_baseline.json"
CKPT_JSONL = RESULTS_DIR / "e1_inverter_eval_v3.ckpt.jsonl"


def parse_latent(z_raw: str | dict) -> dict:
    if isinstance(z_raw, str):
        return json.loads(z_raw)
    return z_raw


def ece_from_preds(y_all: np.ndarray, p_all: np.ndarray, n_bins: int = 10) -> float:
    """Compute expected calibration error."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_all)
    for i in range(n_bins):
        mask = (p_all >= bins[i]) & (p_all < bins[i + 1])
        if mask.sum() == 0:
            continue
        acc = y_all[mask].mean()
        conf = p_all[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def eval_prefix(records: list[dict], kc_ids: list[str]) -> dict:
    """Compute KC Brier (continuous), ECE, AUC, and misc MRR for a set of records.

    KC Brier uses continuous silver probabilities as targets (matching main eval).
    KC AUC uses binarized silver > 0.5 for ranking.
    """
    # Continuous targets for Brier
    g_cont_all, p_all = [], []
    # Binary targets for AUC
    per_kc_y: dict[str, list] = {kc: [] for kc in kc_ids}
    per_kc_p: dict[str, list] = {kc: [] for kc in kc_ids}
    y_bin_all, p_bin_all = [], []
    misc_mrr_vals = []

    for rec in records:
        pred = parse_latent(rec["z_hat"])
        gold = parse_latent(rec["z_gold"])
        pred_m = pred.get("mastery", {}).get("values", {})
        gold_m = gold.get("mastery", {}).get("values", {})
        pred_misc = pred.get("misconceptions", {}).get("probs", {})
        gold_misc = gold.get("misconceptions", {}).get("probs", {})

        for kc in kc_ids:
            g_cont = float(gold_m.get(kc, 0.5))
            p = float(pred_m.get(kc, 0.5))
            g_bin = int(g_cont > 0.5)
            g_cont_all.append(g_cont)
            p_all.append(p)
            y_bin_all.append(g_bin)
            p_bin_all.append(p)
            per_kc_y[kc].append(g_bin)
            per_kc_p[kc].append(p)

        # MRR for misconception ranking
        gold_active = {m for m, v in gold_misc.items() if v > 0.3}
        ranked = sorted(pred_misc, key=lambda m: pred_misc[m], reverse=True)
        mrr_val = 0.0
        for i, m in enumerate(ranked[:20]):
            if m in gold_active:
                mrr_val = 1.0 / (i + 1)
                break
        if gold_active:
            misc_mrr_vals.append(mrr_val)

    g_arr = np.array(g_cont_all)
    p_arr = np.array(p_all)
    y_bin_arr = np.array(y_bin_all)

    # Continuous Brier (matches main eval)
    kc_brier = float(np.mean((p_arr - g_arr) ** 2))
    kc_ece = ece_from_preds(y_bin_arr, p_arr)

    # Per-KC AUC (only where binary labels vary)
    per_kc_aucs = []
    for kc in kc_ids:
        y = np.array(per_kc_y[kc])
        p = np.array(per_kc_p[kc])
        if 0 < y.sum() < len(y):
            try:
                per_kc_aucs.append(roc_auc_score(y, p))
            except Exception:
                pass

    # Pooled AUC
    pooled_auc = float("nan")
    if 0 < y_bin_arr.sum() < len(y_bin_arr):
        try:
            pooled_auc = float(roc_auc_score(y_bin_arr, p_arr))
        except Exception:
            pass

    macro_per_kc_auc = float(np.mean(per_kc_aucs)) if per_kc_aucs else float("nan")
    misc_mrr = float(np.mean(misc_mrr_vals)) if misc_mrr_vals else float("nan")

    return {
        "n": len(records),
        "kc_brier": round(kc_brier, 4),
        "kc_ece": round(kc_ece, 4),
        "pooled_kc_auc": round(pooled_auc, 4) if not math.isnan(pooled_auc) else "N/A",
        "macro_per_kc_auc": round(macro_per_kc_auc, 4) if not math.isnan(macro_per_kc_auc) else "N/A",
        "misc_mrr": round(misc_mrr, 4) if not math.isnan(misc_mrr) else "N/A",
    }


def main() -> None:
    kc_ids = [f"KC{i:02d}" for i in range(1, 31)]

    if not CKPT_JSONL.exists():
        print(f"[err] Missing {CKPT_JSONL}")
        return

    # Load all records by prefix
    by_prefix: dict[str, list[dict]] = {}
    for line in CKPT_JSONL.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("fail"):
            continue
        # Normalize key: ckpt uses 'did' not 'dialogue_id'
        if "did" in rec and "dialogue_id" not in rec:
            rec["dialogue_id"] = rec["did"]
        by_prefix.setdefault(rec["prefix"], []).append(rec)

    print(f"Loaded prefixes: {sorted(by_prefix.keys())}")
    prefix_labels = {
        "T2": "T=2 (problem + 1 student turn)",
        "T4": "T=4 (problem + 2 exchanges)",
        "T8": "T=8 (problem + 4 exchanges)",
        "Tfull": "Full dialogue",
    }

    # Evaluate each prefix
    prefix_results: dict[str, dict] = {}
    for prefix, label in prefix_labels.items():
        if prefix in by_prefix:
            metrics = eval_prefix(by_prefix[prefix], kc_ids)
            metrics["label"] = label
            prefix_results[prefix] = metrics
        else:
            print(f"[warn] Prefix {prefix} not found in checkpoint")

    # Pull pseudo-Z (prior), BKT, gpt-4o from existing result files
    external: dict[str, dict] = {}

    if INV_V3_JSON.exists():
        inv = json.loads(INV_V3_JSON.read_text(encoding="utf-8"))["results"]
        pseudo = inv.get("pseudo", {})
        external["pseudo_z_prior"] = {
            "label": "Pseudo-Z (prior, no dialogue)",
            "n": pseudo.get("n", "?"),
            "kc_brier": round(pseudo.get("kc_brier_mean", float("nan")), 4),
            "pooled_kc_auc": round(pseudo.get("kc_auc_mean", float("nan")), 4),
            "misc_mrr": round(pseudo.get("misc_mrr_at_10", float("nan")), 4),
            "kc_ece": "N/A",
            "macro_per_kc_auc": "N/A",
        }
        bkt = inv.get("bkt", {})
        external["bkt"] = {
            "label": "BKT (KC sequence only)",
            "n": bkt.get("n", "?"),
            "kc_brier": round(bkt.get("kc_brier_mean", float("nan")), 4),
            "pooled_kc_auc": "N/A",
            "misc_mrr": "N/A",
            "kc_ece": "N/A",
            "macro_per_kc_auc": "N/A",
        }

    if GPT4O_JSON.exists():
        gpt = json.loads(GPT4O_JSON.read_text(encoding="utf-8"))
        gpt_metrics = gpt.get("results", gpt)
        gpt_tfull = gpt_metrics.get("gpt4o_zs", gpt_metrics.get("Tfull", gpt_metrics.get("full", {})))
        external["gpt4o_zs"] = {
            "label": "GPT-4o zero-shot (full dialogue)",
            "n": gpt_tfull.get("n", "?"),
            "kc_brier": round(gpt_tfull.get("kc_brier_mean", float("nan")), 4),
            "pooled_kc_auc": round(gpt_tfull.get("kc_auc_mean", float("nan")), 4),
            "misc_mrr": round(gpt_tfull.get("misc_mrr_at_10", float("nan")), 4),
            "kc_ece": "N/A",
            "macro_per_kc_auc": "N/A",
        }

    # Extended analysis for macro per-KC AUC
    if EXTENDED_JSON.exists():
        ext = json.loads(EXTENDED_JSON.read_text(encoding="utf-8"))
        for prefix in ["T2", "T4", "T8", "Tfull"]:
            if prefix in prefix_results and prefix in ext:
                perkc = ext[prefix].get("per_kc_auc", {})
                if perkc:
                    macro = float(np.mean(list(perkc.values())))
                    prefix_results[prefix]["macro_per_kc_auc"] = round(macro, 4)
                ece_val = ext[prefix].get("kc_ece")
                if ece_val is not None:
                    prefix_results[prefix]["kc_ece"] = round(ece_val, 4)

    # Combine results
    all_results = {**external, **{f"ISS_{p}": v for p, v in prefix_results.items()}}

    # Compute delta: does full dialogue improve over T2 baseline?
    delta_brier_t2_vs_full = "N/A"
    delta_auc_t2_vs_full = "N/A"
    if "T2" in prefix_results and "Tfull" in prefix_results:
        t2_brier = prefix_results["T2"]["kc_brier"]
        full_brier = prefix_results["Tfull"]["kc_brier"]
        delta_brier_t2_vs_full = round(t2_brier - full_brier, 4)
        t2_auc = prefix_results["T2"].get("pooled_kc_auc", float("nan"))
        full_auc = prefix_results["Tfull"].get("pooled_kc_auc", float("nan"))
        if not isinstance(t2_auc, str) and not isinstance(full_auc, str):
            delta_auc_t2_vs_full = round(float(full_auc) - float(t2_auc), 4)

    results = {
        "summary": all_results,
        "delta_analysis": {
            "brier_improvement_t2_to_tfull": delta_brier_t2_vs_full,
            "auc_improvement_t2_to_tfull": delta_auc_t2_vs_full,
            "interpretation": (
                "A positive Brier delta indicates that full dialogue reduces calibration error "
                "relative to the T=2 (problem-only) condition. A positive AUC delta indicates "
                "that full dialogue improves KC discrimination. If both deltas are near zero, "
                "the inverter does not effectively leverage student dialogue evidence beyond "
                "what is recoverable from the problem statement alone."
            ),
        },
    }

    out_json = RESULTS_DIR / "robustness_dialogue_baselines.json"
    out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[ok] Wrote {out_json}")

    # Markdown
    md_lines = [
        "# Dialogue Evidence Baselines",
        "",
        "Compares ISS inverter at increasing dialogue prefix lengths against "
        "non-dialogue baselines (pseudo-Z prior, BKT, GPT-4o ZS).",
        "",
        "## Main Comparison Table",
        "",
        "| Condition | n | KC Brier↓ | KC ECE↓ | Pooled AUC↑ | Macro per-KC AUC↑ | Misc MRR↑ |",
        "|-----------|---|-----------|---------|------------|-------------------|----------|",
    ]
    row_order = [
        ("pseudo_z_prior", "Pseudo-Z (prior)"),
        ("bkt", "BKT"),
        ("gpt4o_zs", "GPT-4o ZS (full)"),
        ("ISS_T2", "ISS T=2"),
        ("ISS_T4", "ISS T=4"),
        ("ISS_T8", "ISS T=8"),
        ("ISS_Tfull", "ISS Full"),
    ]
    for key, display in row_order:
        if key in all_results:
            r = all_results[key]
            md_lines.append(
                f"| {display} | {r.get('n','?')} "
                f"| {r.get('kc_brier','N/A')} "
                f"| {r.get('kc_ece','N/A')} "
                f"| {r.get('pooled_kc_auc','N/A')} "
                f"| {r.get('macro_per_kc_auc','N/A')} "
                f"| {r.get('misc_mrr','N/A')} |"
            )

    da = results["delta_analysis"]
    md_lines += [
        "",
        "## Delta Analysis: Gain from Full Dialogue vs T=2",
        "",
        f"| Metric | Δ (Tfull - T2) |",
        f"|--------|---------------|",
        f"| KC Brier improvement | {da['brier_improvement_t2_to_tfull']} |",
        f"| Pooled AUC improvement | {da['auc_improvement_t2_to_tfull']} |",
        "",
        "## Interpretation",
        "",
        da["interpretation"],
    ]

    out_md = RESULTS_DIR / "robustness_dialogue_baselines.md"
    out_md.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"[ok] Wrote {out_md}")

    # Print summary table
    print("\nCondition | KC Brier | Pooled AUC | Misc MRR")
    print("-" * 60)
    for key, display in row_order:
        if key in all_results:
            r = all_results[key]
            print(f"{display:30s} | {r.get('kc_brier','N/A'):>8} | {r.get('pooled_kc_auc','N/A'):>10} | {r.get('misc_mrr','N/A'):>8}")

    print(f"\nBrier improvement T2→Tfull: {da['brier_improvement_t2_to_tfull']}")
    print(f"AUC improvement T2→Tfull:   {da['auc_improvement_t2_to_tfull']}")


if __name__ == "__main__":
    main()
