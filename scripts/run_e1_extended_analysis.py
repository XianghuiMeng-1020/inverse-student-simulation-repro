"""E1 extended: bootstrap CI, per-KC AUC, ECE calibration, misc-on-active subset."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import typer
from sklearn.metrics import roc_auc_score

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

from iss.eval.bootstrap import bootstrap_ci
from iss.eval.metrics import binary_auc, binary_brier, ece
from iss.schema.grammar import validate_latent_z_json
from iss.schema.kc_ontology import get_kc_ids
from iss.schema.misconception_catalogue import get_misconception_ids

app = typer.Typer(no_args_is_help=True)
RESULTS = repo_root / "experiments" / "results"
FIGURES = repo_root / "paper" / "figures"


@app.command()
def main(
    ckpt: Path = typer.Option(
        Path("experiments/results/e1_inverter_eval_v3.ckpt.jsonl"),
        "--ckpt",
    ),
    out_json: Path = typer.Option(Path("experiments/results/e1_extended_v3.json"), "--out-json"),
) -> None:
    ckpt_path = ckpt if ckpt.is_absolute() else repo_root / ckpt
    if not ckpt_path.is_file():
        typer.echo(f"[err] missing checkpoint {ckpt_path}", err=True)
        raise typer.Exit(code=1)

    by_prefix: dict[str, list[dict]] = {}
    for line in ckpt_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("fail"):
            continue
        by_prefix.setdefault(row["prefix"], []).append(row)

    report: dict = {}
    kc_ids = get_kc_ids()
    misc_ids = get_misconception_ids()

    for prefix, rows in by_prefix.items():
        preds = [validate_latent_z_json(r["z_hat"]) for r in rows]
        golds = [validate_latent_z_json(r["z_gold"]) for r in rows]
        n = len(preds)

        # KC pooled for bootstrap
        y_all: list[float] = []
        p_all: list[float] = []
        per_kc_auc: dict[str, float] = {}
        for kc in kc_ids:
            y = np.array([(g.mastery.values[kc] > 0.5) for g in golds], dtype=int)
            p = np.array([pr.mastery.values[kc] for pr in preds], dtype=float)
            if 0 < y.sum() < len(y):
                per_kc_auc[kc] = float(roc_auc_score(y, p))
            y_all.extend(y.tolist())
            p_all.extend(p.tolist())

        kc_brier_ci = bootstrap_ci(
            np.array(y_all),
            np.array(p_all),
            lambda y, p: float(np.mean((p - y) ** 2)),
            n_boot=1000,
            seed=42,
        )
        kc_auc_ci = bootstrap_ci(
            np.array(y_all),
            np.array(p_all),
            binary_auc,
            n_boot=1000,
            seed=43,
        )

        # Misc on dialogues with any gold misc >= 0.01
        active_idx = [
            i
            for i, g in enumerate(golds)
            if any(g.misconceptions.probs[m] >= 0.01 for m in misc_ids)
        ]
        misc_f1_active = float("nan")
        if active_idx:
            f1s = []
            for i in active_idx:
                pred_sorted = sorted(
                    misc_ids, key=lambda m: preds[i].misconceptions.probs.get(m, 0.0), reverse=True
                )
                gold_sorted = sorted(
                    misc_ids, key=lambda m: golds[i].misconceptions.probs.get(m, 0.0), reverse=True
                )
                gt5 = set(gold_sorted[:5])
                pt5 = set(pred_sorted[:5])
                tp = len(pt5 & gt5)
                p, r = tp / 5, tp / max(len(gt5), 1)
                f1s.append(2 * p * r / (p + r) if (p + r) > 0 else 0.0)
            misc_f1_active = float(np.mean(f1s))

        # ECE on pooled KC binary
        ece_val = ece(p_all, [int(x) for x in y_all], n_bins=15)

        report[prefix] = {
            "n": n,
            "n_misc_active": len(active_idx),
            "kc_brier_ci": kc_brier_ci,
            "kc_auc_ci": kc_auc_ci,
            "misc_f1_at_5_active_only": misc_f1_active,
            "kc_ece": ece_val,
            "per_kc_auc": per_kc_auc,
        }

    out_path = out_json if out_json.is_absolute() else repo_root / out_json
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Per-KC heatmap (Tfull or largest prefix)
    best_key = "Tfull" if "Tfull" in report else next(iter(report))
    aucs = report[best_key]["per_kc_auc"]
    if aucs:
        FIGURES.mkdir(parents=True, exist_ok=True)
        keys = sorted(aucs.keys())
        vals = [aucs[k] for k in keys]
        fig, ax = plt.subplots(figsize=(10, 2.5))
        im = ax.imshow([vals], aspect="auto", cmap="RdYlGn", vmin=0.4, vmax=0.9)
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels(keys, rotation=90, fontsize=6)
        ax.set_yticks([0])
        ax.set_yticklabels(["AUC"])
        plt.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(FIGURES / "fig5_kc_auc_heatmap.pdf")
        plt.close(fig)

    # ECE reliability diagram
    if p_all:
        fig, ax = plt.subplots(figsize=(4, 4))
        bins = np.linspace(0, 1, 16)
        bin_centers = []
        bin_acc = []
        p_arr = np.array(p_all)
        y_arr = np.array(y_all)
        for lo, hi in zip(bins[:-1], bins[1:], strict=True):
            m = (p_arr >= lo) & (p_arr < hi)
            if m.sum() == 0:
                continue
            bin_centers.append((lo + hi) / 2)
            bin_acc.append(y_arr[m].mean())
        ax.plot([0, 1], [0, 1], "k--", label="perfect")
        ax.plot(bin_centers, bin_acc, "o-", label="model")
        ax.set_xlabel("Predicted P(mastery)")
        ax.set_ylabel("Empirical frequency")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIGURES / "fig4_ece_reliability.pdf")
        plt.close(fig)

    typer.echo(f"[done] -> {out_path}")


if __name__ == "__main__":
    app()
