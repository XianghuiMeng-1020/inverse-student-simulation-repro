"""Wilcoxon signed-rank and Cohen's d for E4 replay condition pairs."""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import typer
from scipy.stats import wilcoxon

repo_root = Path(__file__).resolve().parents[1]
app = typer.Typer(no_args_is_help=True)


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    diff = x - y
    if diff.std() < 1e-12:
        return 0.0
    return float(diff.mean() / diff.std())


@app.command()
def main(
    replay_json: Path = typer.Option(Path("experiments/results/e4_oracle_v3.json"), "--replay-json"),
    out_json: Path = typer.Option(Path("experiments/results/e4_statistics.json"), "--out-json"),
) -> None:
    path = replay_json if replay_json.is_absolute() else repo_root / replay_json
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("rows", data)

    pairs = [
        ("nll_mean_pseudo", "nll_mean_random_z", "pseudo_vs_random"),
        ("nll_mean_silver_full_dialogue_z", "nll_mean_random_z", "silver_vs_random"),
        ("nll_mean_silver_full_dialogue_z", "nll_mean_pseudo", "silver_vs_pseudo"),
    ]

    by_h: dict[int, list] = defaultdict(list)
    for r in rows:
        by_h[int(r.get("horizon", 1))].append(r)

    report: dict = {}
    for h, rs in sorted(by_h.items()):
        report[str(h)] = {}
        for ka, kb, name in pairs:
            a = np.array([float(r[ka]) for r in rs if r.get(ka) is not None and r.get(kb) is not None])
            b = np.array([float(r[kb]) for r in rs if r.get(ka) is not None and r.get(kb) is not None])
            if len(a) < 10:
                continue
            stat, p = wilcoxon(a, b, alternative="two-sided")
            report[str(h)][name] = {
                "wilcoxon_stat": float(stat),
                "p_value": float(p),
                "cohens_d": cohens_d(a, b),
                "mean_delta_nll": float((a - b).mean()),
                "mean_ppl_a": float(np.exp(a.mean())),
                "mean_ppl_b": float(np.exp(b.mean())),
            }

    out_path = out_json if out_json.is_absolute() else repo_root / out_json
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    typer.echo(json.dumps(report, indent=2))


if __name__ == "__main__":
    app()
