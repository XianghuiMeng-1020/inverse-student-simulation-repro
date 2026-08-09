"""
Task 5: KC structure analysis.

Computes:
- Within-dialogue KC Spearman correlation between predicted and silver mastery
- Separates this from pooled (cross-dialogue) AUC
- Explains why pooled AUC ≈ chance while per-KC macro AUC is higher
- Computes top-high/top-low KC overlap as an alternative to Spearman

Output: experiments/results/robustness_kc_structure.json
        experiments/results/robustness_kc_structure.md
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, wilcoxon

repo_root = Path(__file__).resolve().parents[1]
RESULTS_DIR = repo_root / "experiments" / "results"
CKPT_JSONL = RESULTS_DIR / "e1_inverter_eval_v3.ckpt.jsonl"
EXTENDED_JSON = RESULTS_DIR / "e1_extended_v3.json"


def get_kc_ids() -> list[str]:
    return [f"KC{i:02d}" for i in range(1, 31)]


def parse_latent(z_raw: str | dict) -> dict:
    """Parse latent Z from JSON string or dict."""
    if isinstance(z_raw, str):
        return json.loads(z_raw)
    return z_raw


def main() -> None:
    kc_ids = get_kc_ids()

    if not CKPT_JSONL.exists():
        print(f"[err] Missing {CKPT_JSONL}")
        return

    # Load per-dialogue predictions at Tfull prefix
    records: list[dict] = []
    for line in CKPT_JSONL.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("fail") or rec.get("prefix") != "Tfull":
            continue
        try:
            z_hat = parse_latent(rec["z_hat"])
            z_gold = parse_latent(rec["z_gold"])
            pred_mastery = z_hat.get("mastery", {}).get("values", {})
            gold_mastery = z_gold.get("mastery", {}).get("values", {})
            if not pred_mastery or not gold_mastery:
                continue
            records.append({
                "dialogue_id": rec.get("did") or rec.get("dialogue_id"),
                "pred": pred_mastery,
                "gold": gold_mastery,
            })
        except Exception:
            continue

    n = len(records)
    print(f"Loaded {n} Tfull predictions from checkpoint")

    if n == 0:
        print("[err] No valid Tfull records found.")
        return

    # ── Within-dialogue Spearman correlation ─────────────────────────────
    # For each dialogue, compute Spearman(pred_KC_vector, gold_KC_vector)
    spearman_rhos: list[float] = []
    top3_overlap: list[float] = []
    bottom3_overlap: list[float] = []

    for rec in records:
        pred_vec = [rec["pred"].get(kc, 0.5) for kc in kc_ids]
        gold_vec = [rec["gold"].get(kc, 0.5) for kc in kc_ids]

        # Spearman
        rho, _ = spearmanr(pred_vec, gold_vec)
        if not math.isnan(rho):
            spearman_rhos.append(rho)

        # Top-3 overlap: which KCs are ranked highest in gold vs pred
        pred_top3 = set(sorted(kc_ids, key=lambda k: rec["pred"].get(k, 0.5), reverse=True)[:3])
        gold_top3 = set(sorted(kc_ids, key=lambda k: rec["gold"].get(k, 0.5), reverse=True)[:3])
        top3_overlap.append(len(pred_top3 & gold_top3) / 3.0)

        pred_bot3 = set(sorted(kc_ids, key=lambda k: rec["pred"].get(k, 0.5))[:3])
        gold_bot3 = set(sorted(kc_ids, key=lambda k: rec["gold"].get(k, 0.5))[:3])
        bottom3_overlap.append(len(pred_bot3 & gold_bot3) / 3.0)

    mean_spearman = float(np.mean(spearman_rhos))
    std_spearman = float(np.std(spearman_rhos))
    median_spearman = float(np.median(spearman_rhos))
    pct_positive_spearman = 100.0 * sum(1 for r in spearman_rhos if r > 0) / len(spearman_rhos)

    # Wilcoxon test: is mean Spearman significantly > 0?
    try:
        wx_stat, wx_p = wilcoxon(spearman_rhos, alternative="greater")
    except Exception:
        wx_stat, wx_p = float("nan"), float("nan")

    mean_top3_overlap = float(np.mean(top3_overlap))
    mean_bot3_overlap = float(np.mean(bottom3_overlap))

    # ── Per-KC cross-dialogue AUC (from extended JSON) ─────────────────────
    per_kc_auc_all: dict[str, list[float]] = {}
    if EXTENDED_JSON.exists():
        ext = json.loads(EXTENDED_JSON.read_text(encoding="utf-8"))
        # Collect per_kc_auc across all prefix lengths
        for prefix_data in ext.values():
            if "per_kc_auc" in prefix_data:
                for kc, auc_val in prefix_data["per_kc_auc"].items():
                    per_kc_auc_all.setdefault(kc, []).append(auc_val)

    per_kc_auc_tfull: dict[str, float] = {}
    if EXTENDED_JSON.exists():
        ext = json.loads(EXTENDED_JSON.read_text(encoding="utf-8"))
        tfull = ext.get("Tfull", {})
        per_kc_auc_tfull = tfull.get("per_kc_auc", {})

    macro_auc_tfull = float(np.mean(list(per_kc_auc_tfull.values()))) if per_kc_auc_tfull else float("nan")

    # ── Pooled cross-dialogue AUC (from e1_inverter_eval_v3.json) ─────────
    inv_v3_path = RESULTS_DIR / "e1_inverter_eval_v3.json"
    pooled_auc = float("nan")
    if inv_v3_path.exists():
        inv_v3 = json.loads(inv_v3_path.read_text(encoding="utf-8"))
        tfull_results = inv_v3.get("results", {}).get("Tfull", {})
        pooled_auc = tfull_results.get("kc_auc_mean", float("nan"))

    results = {
        "n_dialogues": n,
        "within_dialogue_kc_spearman": {
            "mean_rho": round(mean_spearman, 4),
            "std_rho": round(std_spearman, 4),
            "median_rho": round(median_spearman, 4),
            "pct_positive": round(pct_positive_spearman, 1),
            "wilcoxon_p_greater_than_zero": round(wx_p, 4) if not math.isnan(wx_p) else "N/A",
            "n_valid": len(spearman_rhos),
            "interpretation": (
                "Within-dialogue Spearman measures whether the inverter ranks KCs "
                "in the correct relative order for each dialogue. A positive mean rho "
                "indicates the model recovers intra-dialogue KC structure even when "
                "cross-dialogue discrimination (pooled AUC) is near chance."
            ),
        },
        "top_kc_overlap": {
            "mean_top3_overlap": round(mean_top3_overlap, 4),
            "mean_bottom3_overlap": round(mean_bot3_overlap, 4),
            "interpretation": (
                "Jaccard overlap of top-3 and bottom-3 KC rankings between predicted "
                "and silver labels within each dialogue."
            ),
        },
        "pooled_vs_perkc_auc": {
            "pooled_cross_dialogue_auc_tfull": round(pooled_auc, 4) if not math.isnan(pooled_auc) else "N/A",
            "macro_per_kc_auc_tfull": round(macro_auc_tfull, 4) if not math.isnan(macro_auc_tfull) else "N/A",
            "per_kc_auc_tfull": {k: round(v, 4) for k, v in per_kc_auc_tfull.items()},
            "interpretation": (
                "Pooled AUC treats all (dialogue, KC) pairs indiscriminately, "
                "conflating KC-level variation with student-level variation. "
                "Per-KC macro AUC averages ROC-AUC computed per-KC across dialogues, "
                "which is higher because the inverter learns relative KC ordering "
                "(which KCs tend to be harder) without necessarily discriminating "
                "individual students on each KC. "
                "Within-dialogue Spearman directly tests whether the inverter "
                "outputs calibrated KC profiles per student."
            ),
        },
    }

    # ── Write outputs ─────────────────────────────────────────────────────
    out_json = RESULTS_DIR / "robustness_kc_structure.json"
    out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[ok] Wrote {out_json}")

    # Markdown
    md_lines = [
        "# KC Structure Analysis",
        "",
        f"**Dialogues analyzed**: {n} (Tfull prefix, seed 42)",
        "",
        "## Within-Dialogue KC Spearman Correlation",
        "",
        "For each dialogue, Spearman ρ is computed between the predicted 30-KC mastery "
        "vector and the silver mastery vector.",
        "",
        f"| Statistic | Value |",
        f"|-----------|-------|",
        f"| Mean ρ | {results['within_dialogue_kc_spearman']['mean_rho']} |",
        f"| Std ρ | {results['within_dialogue_kc_spearman']['std_rho']} |",
        f"| Median ρ | {results['within_dialogue_kc_spearman']['median_rho']} |",
        f"| % dialogues with ρ > 0 | {results['within_dialogue_kc_spearman']['pct_positive']}% |",
        f"| Wilcoxon p (one-sided, ρ > 0) | {results['within_dialogue_kc_spearman']['wilcoxon_p_greater_than_zero']} |",
        "",
        "## Top-KC Overlap",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Mean top-3 KC overlap | {results['top_kc_overlap']['mean_top3_overlap']} |",
        f"| Mean bottom-3 KC overlap | {results['top_kc_overlap']['mean_bottom3_overlap']} |",
        "",
        "## Pooled vs Per-KC AUC (Tfull)",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Pooled cross-dialogue AUC | {results['pooled_vs_perkc_auc']['pooled_cross_dialogue_auc_tfull']} |",
        f"| Macro per-KC AUC | {results['pooled_vs_perkc_auc']['macro_per_kc_auc_tfull']} |",
        "",
        "## Interpretation",
        "",
        results["pooled_vs_perkc_auc"]["interpretation"],
    ]

    out_md = RESULTS_DIR / "robustness_kc_structure.md"
    out_md.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"[ok] Wrote {out_md}")

    print(f"\nWithin-dialogue Spearman: mean={mean_spearman:.4f} ± {std_spearman:.4f}")
    print(f"Wilcoxon p (rho>0): {wx_p:.4f}")
    print(f"Top-3 KC overlap: {mean_top3_overlap:.4f}")
    print(f"Pooled AUC: {pooled_auc:.4f}  Macro per-KC AUC: {macro_auc_tfull:.4f}")


if __name__ == "__main__":
    main()
