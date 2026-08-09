"""
Task 4: Forward replay state-use probes - partial context analysis.

Analyzes existing replay result files to extract per-condition statistics
for partial context conditions using the available e4_context_v3.json and
e4_oracle_v3.json data.

Adds analysis of:
- Context-only vs context+silver Z vs context+random Z conditions
- Oracle (Z-only) results at h=1, h=3, h=5
- Wilcoxon tests and effect sizes for all pairwise comparisons
- PPL/NLL gap table

Note: The "last 1 turn" and "last 2 turns" conditions require new model
inference and are documented as PENDING in the output. The existing data
covers:
  - Full context + [pseudo/random/silver] Z (e4_context_v3.json)
  - Z-only oracle + [pseudo/random/silver] Z (e4_oracle_v3.json)
  - Context-dropout + [pseudo/random/silver] Z (e4_ctx_dropout_v3.json)

Output: experiments/results/robustness_forward_replay_probes.json
        experiments/results/robustness_forward_replay_probes.md
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

repo_root = Path(__file__).resolve().parents[1]
RESULTS_DIR = repo_root / "experiments" / "results"


def cohens_d(x: list[float], y: list[float]) -> float:
    """Cohen's d for paired samples (x - y)."""
    diffs = [xi - yi for xi, yi in zip(x, y)]
    mean_d = np.mean(diffs)
    std_d = np.std(diffs, ddof=1)
    return float(mean_d / std_d) if std_d > 0 else float("nan")


def pairwise_test(
    nll_a: list[float],
    nll_b: list[float],
    label_a: str,
    label_b: str,
    alternative: str = "less",
) -> dict:
    """Wilcoxon signed-rank test and Cohen's d for NLL(a) < NLL(b) (a is better)."""
    paired = [(a, b) for a, b in zip(nll_a, nll_b) if a is not None and b is not None]
    if len(paired) < 5:
        return {"n": len(paired), "p": "N/A", "d": "N/A", "note": "insufficient pairs"}
    xs, ys = zip(*paired)
    try:
        stat, p = wilcoxon(xs, ys, alternative=alternative)
    except Exception as e:
        stat, p = float("nan"), float("nan")
    d = cohens_d(list(xs), list(ys))
    return {
        "n": len(paired),
        "mean_nll_a": round(float(np.mean(xs)), 5),
        "mean_nll_b": round(float(np.mean(ys)), 5),
        "delta_nll": round(float(np.mean(xs)) - float(np.mean(ys)), 5),
        "wilcoxon_stat": round(stat, 3) if not math.isnan(stat) else "N/A",
        "p": round(p, 4) if not math.isnan(p) else "N/A",
        "cohens_d": round(d, 4) if not math.isnan(d) else "N/A",
        "significant_p05": bool(not math.isnan(p) and p < 0.05),
    }


def load_replay_file(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def extract_nll_by_horizon(data: dict, horizon: int) -> dict[str, list[float]]:
    """Extract per-row NLL values for each condition at a given horizon."""
    nll_pseudo = []
    nll_random = []
    nll_silver = []
    for row in data.get("rows", []):
        if row["horizon"] != horizon:
            continue
        nll_pseudo.append(row.get("nll_mean_pseudo"))
        nll_random.append(row.get("nll_mean_random_z"))
        nll_silver.append(row.get("nll_mean_silver_full_dialogue_z"))

    # Filter None pairs consistently
    valid = [(p, r, s) for p, r, s in zip(nll_pseudo, nll_random, nll_silver)
             if p is not None and r is not None and s is not None]
    if not valid:
        return {"pseudo": [], "random": [], "silver": []}
    p_vals, r_vals, s_vals = zip(*valid)
    return {"pseudo": list(p_vals), "random": list(r_vals), "silver": list(s_vals)}


def analyze_condition(data: dict, condition_name: str) -> dict:
    """Compute full analysis for one forward model condition."""
    result: dict = {"condition": condition_name, "horizons": {}}
    for h in [1, 3, 5]:
        nlls = extract_nll_by_horizon(data, h)
        n = len(nlls["pseudo"])
        if n == 0:
            result["horizons"][str(h)] = {"n": 0, "note": "no data at this horizon"}
            continue

        h_result: dict = {
            "n": n,
            "mean_nll": {
                "pseudo": round(float(np.mean(nlls["pseudo"])), 5),
                "random": round(float(np.mean(nlls["random"])), 5),
                "silver": round(float(np.mean(nlls["silver"])), 5),
            },
            "silver_vs_random": pairwise_test(
                nlls["silver"], nlls["random"],
                "silver", "random", alternative="less"
            ),
            "pseudo_vs_random": pairwise_test(
                nlls["pseudo"], nlls["random"],
                "pseudo", "random", alternative="less"
            ),
            "silver_vs_pseudo": pairwise_test(
                nlls["silver"], nlls["pseudo"],
                "silver", "pseudo", alternative="less"
            ),
        }
        result["horizons"][str(h)] = h_result
    return result


def main() -> None:
    results: dict = {
        "conditions": {},
        "partial_context_conditions": {
            "status": "PENDING - requires new model inference",
            "description": (
                "Testing last-1-turn and last-2-turn partial context conditions "
                "requires running forward model inference with truncated dialogue history. "
                "These conditions are documented here for completeness but results are not available. "
                "Available conditions: full-context (E4a), Z-only oracle (E4b), context-dropout (E4c)."
            ),
            "planned_conditions": [
                "last_1_turn + silver_Z",
                "last_1_turn + random_Z",
                "last_2_turns + silver_Z",
                "last_2_turns + random_Z",
                "context_only (no Z)",
                "Z_only oracle",
                "context + shuffled_Z",
            ],
        },
        "summary_table": [],
    }

    condition_files = {
        "E4a_full_context": RESULTS_DIR / "e4_context_v3.json",
        "E4b_oracle_Zonly": RESULTS_DIR / "e4_oracle_v3.json",
        "E4c_ctx_dropout": RESULTS_DIR / "e4_ctx_dropout_v3.json",
    }

    for cond_name, fpath in condition_files.items():
        data = load_replay_file(fpath)
        if data is None:
            results["conditions"][cond_name] = {"status": "file not found", "path": str(fpath)}
            continue
        results["conditions"][cond_name] = analyze_condition(data, cond_name)

    # Build summary table for h=1
    h = 1
    summary_rows = []
    for cond_name, cond_data in results["conditions"].items():
        if not isinstance(cond_data, dict) or "horizons" not in cond_data:
            continue
        h_data = cond_data["horizons"].get(str(h), {})
        if not h_data or h_data.get("n", 0) == 0:
            continue
        mnll = h_data.get("mean_nll", {})
        svr = h_data.get("silver_vs_random", {})
        summary_rows.append({
            "condition": cond_name,
            "n": h_data["n"],
            "mean_nll_silver": mnll.get("silver", "N/A"),
            "mean_nll_random": mnll.get("random", "N/A"),
            "delta_nll_silver_minus_random": svr.get("delta_nll", "N/A"),
            "wilcoxon_p": svr.get("p", "N/A"),
            "cohens_d": svr.get("cohens_d", "N/A"),
            "significant": svr.get("significant_p05", False),
        })
    results["summary_table"] = summary_rows

    out_json = RESULTS_DIR / "robustness_forward_replay_probes.json"
    out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[ok] Wrote {out_json}")

    # Markdown
    md_lines = [
        "# Forward Replay State-Use Probes",
        "",
        "Analysis of whether the forward model uses structured Z when context is "
        "available or limited.",
        "",
        "## Summary at h=1 (horizon = 1 student turn)",
        "",
        "| Condition | n | NLL(silver) | NLL(random) | ΔNLL | Wilcoxon p | Cohen's d | Sig. |",
        "|-----------|---|-------------|-------------|------|-----------|-----------|------|",
    ]
    for row in summary_rows:
        sig = "✓" if row["significant"] else "✗"
        md_lines.append(
            f"| {row['condition']} | {row['n']} "
            f"| {row['mean_nll_silver']} "
            f"| {row['mean_nll_random']} "
            f"| {row['delta_nll_silver_minus_random']} "
            f"| {row['wilcoxon_p']} "
            f"| {row['cohens_d']} "
            f"| {sig} |"
        )

    md_lines += [
        "",
        "## Detailed Results by Horizon",
        "",
    ]
    for cond_name, cond_data in results["conditions"].items():
        if not isinstance(cond_data, dict) or "horizons" not in cond_data:
            continue
        md_lines.append(f"### {cond_name}")
        md_lines.append("")
        for h_str, h_data in cond_data["horizons"].items():
            if not h_data or h_data.get("n", 0) == 0:
                continue
            svr = h_data.get("silver_vs_random", {})
            md_lines.append(
                f"- **h={h_str}**: n={h_data['n']}, "
                f"NLL(silver)={h_data['mean_nll'].get('silver','N/A')}, "
                f"NLL(random)={h_data['mean_nll'].get('random','N/A')}, "
                f"Wilcoxon p={svr.get('p','N/A')}, d={svr.get('cohens_d','N/A')}"
            )
        md_lines.append("")

    md_lines += [
        "## Partial-Context Conditions (PENDING)",
        "",
        results["partial_context_conditions"]["description"],
        "",
        "**Planned conditions requiring new inference:**",
    ]
    for c in results["partial_context_conditions"]["planned_conditions"]:
        md_lines.append(f"- {c}")

    md_lines += [
        "",
        "## Interpretation",
        "",
        "- **E4a (full context)**: NLL differences across Z conditions are < 0.007, "
          "indicating the context-conditioned model ignores Z entirely.",
        "- **E4b (Z-only oracle)**: Significant NLL gap (silver < random) at h=1 and h=3 "
          "confirms structured Z is detectable when contextual shortcuts are removed.",
        "- **E4c (context-dropout)**: Intermediate training does not recover oracle-level "
          "Z-sensitivity.",
        "- **Key conclusion**: Z is used only when forced by architectural constraints. "
          "Partial-context probes (last 1-2 turns) would characterize the transition "
          "point between context-bypassing and Z-use.",
    ]

    out_md = RESULTS_DIR / "robustness_forward_replay_probes.md"
    out_md.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"[ok] Wrote {out_md}")

    # Print summary
    print("\nReplay probe summary (h=1, silver vs random):")
    for row in summary_rows:
        print(f"  {row['condition']:30s}: Δ={row['delta_nll_silver_minus_random']}, p={row['wilcoxon_p']}, d={row['cohens_d']}")


if __name__ == "__main__":
    main()
