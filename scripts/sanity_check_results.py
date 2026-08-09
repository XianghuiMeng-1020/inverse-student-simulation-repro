"""Verify manuscript numbers against experiments/results JSON."""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "experiments" / "results"


def check_e1() -> dict:
    for name in ("e1_inverter_eval_v3.json", "e1_inverter_eval_v2.json"):
        p = RESULTS / name
        if p.is_file():
            data = json.loads(p.read_text(encoding="utf-8"))["results"]
            keys = [k for k in data if k != "pseudo"]
            return {k: data[k] for k in ("pseudo", *keys[:4]) if k in data}
    return {"error": "missing e1 json"}


def check_e4() -> dict:
    p = RESULTS / "e4_mathdial_test_silver.json"
    if not p.is_file():
        return {"error": "missing e4"}
    data = json.loads(p.read_text(encoding="utf-8"))
    rows = data if isinstance(data, list) else data.get("rows", data)
    by_h: dict[int, list] = defaultdict(list)
    for r in rows:
        by_h[int(r.get("horizon", 1))].append(r)
    out = {"n_rows": len(rows), "by_horizon": {}}
    all_pseudo, all_silver = [], []
    for h, rs in sorted(by_h.items()):
        p_p = [r["nll_mean_pseudo"] for r in rs if r.get("nll_mean_pseudo") is not None]
        p_s = [r["nll_mean_silver_full_dialogue_z"] for r in rs if r.get("nll_mean_silver_full_dialogue_z") is not None]
        ppls = {
            "pseudo": math.exp(statistics.mean(p_p)),
            "silver": math.exp(statistics.mean(p_s)),
        }
        out["by_horizon"][h] = {"n": len(rs), "ppl": ppls}
        all_pseudo.extend(p_p)
        all_silver.extend(p_s)
    out["aggregate_delta_ppl"] = math.exp(statistics.mean(all_silver)) - math.exp(statistics.mean(all_pseudo))
    return out


def check_e4_oracle() -> dict:
    p = RESULTS / "e4_oracle_v3.json"
    if not p.is_file():
        return {"status": "pending"}
    data = json.loads(p.read_text(encoding="utf-8"))
    rows = data.get("rows", data)
    return {"n_rows": len(rows), "summary": data.get("summary", {})}


def check_labels_v3() -> dict:
    p = REPO / "data" / "labels" / "latent_z_silver_v3.jsonl"
    if not p.is_file():
        return {"error": "missing v3 labels"}
    n = misc_nz = 0
    kc_vals = []
    for line in p.open(encoding="utf-8"):
        if not line.strip():
            continue
        z = json.loads(line)["latent"]
        n += 1
        kc_vals.extend(float(v) for v in z["mastery"]["values"].values())
        if any(float(v) > 0.01 for v in z["misconceptions"]["probs"].values()):
            misc_nz += 1
    return {
        "rows": n,
        "misc_nonzero_frac": misc_nz / n if n else 0,
        "kc_std": statistics.stdev(kc_vals) if len(kc_vals) > 1 else 0,
    }


def check_e5() -> dict:
    p = RESULTS / "e5_expert_agreement.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {"status": "pending"}


def main() -> None:
    report = {
        "e1": check_e1(),
        "e4": check_e4(),
        "e4_oracle": check_e4_oracle(),
        "labels_v3": check_labels_v3(),
        "e5": check_e5(),
    }
    out = REPO / "paper" / "sanity_report.json"
    out.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    print(json.dumps(report, indent=2, default=float))


if __name__ == "__main__":
    main()
