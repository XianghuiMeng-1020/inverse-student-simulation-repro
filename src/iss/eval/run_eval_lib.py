"""Aggregate evaluation helpers (plan p8-05)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from iss.eval.bootstrap import bootstrap_ci
from iss.eval.metrics import binary_auc, binary_brier, mae, pearson_r


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def summarize_binary(pred_path: Path, gold_path: Path) -> dict[str, float]:
    """Expect JSONL rows with ``y`` and ``p`` fields."""

    preds = load_jsonl(pred_path)
    golds = load_jsonl(gold_path)
    y = [float(r["y"]) for r in golds[: len(preds)]]
    p = [float(r["p"]) for r in preds[: len(y)]]
    return {
        "auc": binary_auc(y, p),
        "brier": binary_brier(y, p),
    }


def summarize_binary_with_ci(
    pred_path: Path,
    gold_path: Path,
    *,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict[str, float]:
    """Same as ``summarize_binary`` plus bootstrap 95% CIs for AUC and Brier."""

    preds = load_jsonl(pred_path)
    golds = load_jsonl(gold_path)
    y = [float(r["y"]) for r in golds[: len(preds)]]
    p = [float(r["p"]) for r in preds[: len(y)]]
    auc_ci = bootstrap_ci(y, p, binary_auc, n_boot=n_boot, seed=seed)
    bri_ci = bootstrap_ci(y, p, binary_brier, n_boot=n_boot, seed=seed + 1)
    return {
        "auc": auc_ci["point"],
        "auc_ci_low": auc_ci["ci_low"],
        "auc_ci_high": auc_ci["ci_high"],
        "brier": bri_ci["point"],
        "brier_ci_low": bri_ci["ci_low"],
        "brier_ci_high": bri_ci["ci_high"],
    }


def summarize_regression(pred_path: Path, gold_path: Path, key: str) -> dict[str, float]:
    preds = load_jsonl(pred_path)
    golds = load_jsonl(gold_path)
    n = min(len(preds), len(golds))
    x = [float(preds[i][key]) for i in range(n)]
    y = [float(golds[i][key]) for i in range(n)]
    return {"pearson_r": pearson_r(x, y), "mae": mae(x, y)}


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
