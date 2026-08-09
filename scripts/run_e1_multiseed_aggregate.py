"""Aggregate E1 results across inverter seeds (mean ± std, merge into e1_inverter_eval_v3.json)."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "experiments" / "results"
SEEDS = (42, 1, 2)
METRICS = ("kc_brier_mean", "kc_auc_mean", "misc_f1_at_5", "misc_mrr_at_10", "metacog_r_mean")
PREFIXES = ("T2", "T4", "T8", "Tfull")


def _nanmean(vals: list[float]) -> float:
    clean = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return float(statistics.mean(clean)) if clean else float("nan")


def _nanstd(vals: list[float]) -> float:
    clean = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return float(statistics.stdev(clean)) if len(clean) > 1 else float("nan")


def main() -> None:
    per_seed: dict[str, dict] = {}
    for seed in SEEDS:
        path = RESULTS / f"e1_inverter_eval_v3_s{seed}.json"
        if not path.is_file() and seed == 42:
            path = RESULTS / "e1_inverter_eval_v3.json"
        if path.is_file():
            per_seed[str(seed)] = json.loads(path.read_text(encoding="utf-8"))["results"]

    if not per_seed:
        raise SystemExit("No per-seed E1 JSON files found.")

    base_key = "42" if "42" in per_seed else next(iter(per_seed))
    merged = dict(per_seed[base_key])
    summary: dict[str, dict] = {}

    for prefix in PREFIXES:
        vals_by_metric: dict[str, list[float]] = {m: [] for m in METRICS}
        for seed_data in per_seed.values():
            if prefix not in seed_data:
                continue
            row = seed_data[prefix]
            for m in METRICS:
                vals_by_metric[m].append(row.get(m, float("nan")))
        if not vals_by_metric["kc_auc_mean"]:
            continue
        summary[prefix] = {
            m: {"mean": _nanmean(vals_by_metric[m]), "std": _nanstd(vals_by_metric[m])}
            for m in METRICS
        }
        if prefix in merged:
            for m in METRICS:
                merged[prefix][f"{m}_mean_seeds"] = summary[prefix][m]["mean"]
                merged[prefix][f"{m}_std_seeds"] = summary[prefix][m]["std"]

    out = {
        "results": merged,
        "multiseed_summary": summary,
        "seeds": list(per_seed.keys()),
    }
    out_path = RESULTS / "e1_inverter_eval_v3_multiseed.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[done] -> {out_path}")


if __name__ == "__main__":
    main()
