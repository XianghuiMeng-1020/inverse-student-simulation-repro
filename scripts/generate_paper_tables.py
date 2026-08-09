"""Generate LaTeX tables for paper/ from experiments/results/*.json."""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "experiments" / "results"
TABLES = REPO / "paper" / "tables"


def _fmt(x: float | None, nd: int = 3) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "---"
    return f"{x:.{nd}f}"


def _fmt_ci(m: dict, key: str, nd: int = 3) -> str:
    if key not in m:
        return "---"
    ci = m.get(f"{key}_ci") or m.get("kc_auc_ci")
    if isinstance(ci, dict) and "point" in ci:
        return f"{_fmt(ci['point'], nd)} [{_fmt(ci['ci_low'], nd)}, {_fmt(ci['ci_high'], nd)}]"
    return _fmt(m.get(key), nd)


def _fmt_pm(mean: float | None, std: float | None, nd: int = 3) -> str:
    if mean is None or (isinstance(mean, float) and math.isnan(mean)):
        return "---"
    if std is None or (isinstance(std, float) and math.isnan(std)):
        return _fmt(mean, nd)
    return f"{mean:.{nd}f} $\\pm$ {std:.{nd}f}"


def write_e1_table() -> None:
    multiseed_path = RESULTS / "e1_inverter_eval_v3_multiseed.json"
    if multiseed_path.is_file():
        ms = json.loads(multiseed_path.read_text(encoding="utf-8"))
        data = ms["results"]
        summary = ms.get("multiseed_summary", {})
        seeds = ms.get("seeds", [])
        caption_suffix = (
            f" ISS inverter rows: mean $\\pm$ std over seeds {', '.join(seeds)}."
            if seeds
            else ""
        )
    else:
        for name in ("e1_inverter_eval_v3_s42.json", "e1_inverter_eval_v3.json", "e1_inverter_eval_v2.json", "e1_inverter_eval.json"):
            path = RESULTS / name
            if path.is_file():
                break
        else:
            return
        data = json.loads(path.read_text(encoding="utf-8"))["results"]
        summary = {}
        caption_suffix = ""
    gpt_path = RESULTS / "e1_gpt4o_zs_v3.json"
    if gpt_path.is_file():
        gpt_res = json.loads(gpt_path.read_text(encoding="utf-8")).get("results", {})
        if "gpt4o_zs" in gpt_res:
            data["gpt4o_zs"] = gpt_res["gpt4o_zs"]
    ext_path = RESULTS / "e1_extended_v3.json"
    ext = json.loads(ext_path.read_text(encoding="utf-8")) if ext_path.is_file() else {}

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Inversion accuracy on MathDial test (silver v3 gold). "
        r"Lower is better for Brier; higher for better AUC/F1/MRR."
        + caption_suffix
        + r"}",
        r"\label{tab:e1-main}",
        r"\small",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Method & $n$ & KC Brier & KC AUC & Misc.\ F1@5 & Misc.\ MRR@10 \\",
        r"\midrule",
    ]
    labels = {
        "pseudo": r"Pseudo-$Z$ heuristic",
        "gpt4o_zs": r"gpt-4o (ZS inverter)",
        "bkt": r"BKT baseline",
        "T2": r"ISS inverter ($T{=}2$)",
        "T4": r"ISS inverter ($T{=}4$)",
        "T8": r"ISS inverter ($T{=}8$)",
        "Tfull": r"ISS inverter (full)",
        "T50": r"ISS inverter ($T_{50}$)",
        "T100": r"ISS inverter ($T_{100}$)",
    }
    order = ["pseudo", "bkt", "gpt4o_zs", "T2", "T4", "T8", "Tfull", "T50", "T100"]
    for key in order:
        if key not in data:
            continue
        m = data[key]
        label = labels.get(key, key)
        sm = summary.get(key, {})
        if sm:
            brier_str = _fmt_pm(sm.get("kc_brier_mean", {}).get("mean"), sm.get("kc_brier_mean", {}).get("std"), 4)
            auc_str = _fmt_pm(sm.get("kc_auc_mean", {}).get("mean"), sm.get("kc_auc_mean", {}).get("std"), 3)
            f1 = sm.get("misc_f1_at_5", {}).get("mean")
            mrr = sm.get("misc_mrr_at_10", {}).get("mean")
        else:
            brier_str = _fmt(m.get("kc_brier_mean"), 4)
            if key in ext and "kc_auc_ci" in ext[key]:
                auc_str = _fmt_ci(ext[key], "kc_auc")
            else:
                auc_str = _fmt(m.get("kc_auc_mean"), 3)
            f1 = m.get("misc_f1_at_5")
            mrr = m.get("misc_mrr_at_10")
        if key in ext and not math.isnan(ext[key].get("misc_f1_at_5_active_only", float("nan"))):
            f1 = ext[key]["misc_f1_at_5_active_only"]
            label += r"$^\dagger$"
        lines.append(
            f"{label} & {m['n']} & {brier_str} & {auc_str} "
            f"& {_fmt(f1, 3)} & {_fmt(mrr, 3)} \\\\"
        )
    lines.extend(
        [
            r"\midrule",
            r"\multicolumn{6}{@{}l@{}}{\footnotesize $^\dagger$ Misc.\ F1@5 on dialogues with $\ge 1$ active misconception in gold.} \\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    (TABLES / "e1_main.tex").write_text("\n".join(lines), encoding="utf-8")


def write_e4_table() -> None:
    for name in ("e4_context_v3.json", "e4_mathdial_test_silver.json"):
        path = RESULTS / name
        if path.is_file():
            break
    else:
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data if isinstance(data, list) else data.get("rows", data)
    by_h: dict[int, list] = defaultdict(list)
    for r in rows:
        by_h[int(r.get("horizon", 1))].append(r)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{E4a: Context-conditioned replay (MathDial test, silver v3 forward). Mean perplexity under three $Z$ assignments.}",
        r"\label{tab:e4-replay}",
        r"\small",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"$h$ & Pseudo-$Z$ & Random-$Z$ & Silver-$Z$ \\",
        r"\midrule",
    ]
    max_gap = 0.0
    for h in sorted(by_h.keys()):
        rs = by_h[h]
        ppls = {}
        for arm, key in [
            ("pseudo", "nll_mean_pseudo"),
            ("random", "nll_mean_random_z"),
            ("silver", "nll_mean_silver_full_dialogue_z"),
        ]:
            vals = [r[key] for r in rs if r.get(key) is not None]
            ppls[arm] = math.exp(statistics.mean(vals)) if vals else float("nan")
        max_gap = max(max_gap, abs(ppls["silver"] - ppls["pseudo"]))
        lines.append(
            f"{h} & {_fmt(ppls['pseudo'], 3)} & {_fmt(ppls['random'], 3)} & {_fmt(ppls['silver'], 3)} \\\\"
        )
    lines.extend(
        [
            r"\midrule",
            rf"\multicolumn{{4}}{{@{{}}l@{{}}}}{{\footnotesize Max pairwise PPL gap $\le {_fmt(max_gap, 3)}$ (context model ignores $Z$).}} \\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    (TABLES / "e4_replay.tex").write_text("\n".join(lines), encoding="utf-8")


def write_e4_oracle_table() -> None:
    path = RESULTS / "e4_oracle_v3.json"
    if not path.is_file():
        (TABLES / "e4_oracle.tex").write_text(
            "% e4_oracle.tex — run E4b after forward_3b_oracle_v3 training\n",
            encoding="utf-8",
        )
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("rows", data)
    by_h: dict[int, list] = defaultdict(list)
    for r in rows:
        by_h[int(r.get("horizon", 1))].append(r)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{E4b: Z-only oracle forward replay. Mean perplexity; larger gaps indicate $Z$ usage.}",
        r"\label{tab:e4-oracle}",
        r"\small",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"$h$ & Silver-$Z$ & Random-$Z$ & Pseudo-$Z$ \\",
        r"\midrule",
    ]
    for h in sorted(by_h.keys()):
        rs = by_h[h]
        ppls = {}
        for arm, key in [
            ("silver", "nll_mean_silver_full_dialogue_z"),
            ("random", "nll_mean_random_z"),
            ("pseudo", "nll_mean_pseudo"),
        ]:
            vals = [r[key] for r in rs if r.get(key) is not None]
            ppls[arm] = math.exp(statistics.mean(vals)) if vals else float("nan")
        lines.append(
            f"{h} & {_fmt(ppls['silver'], 3)} & {_fmt(ppls['random'], 3)} & {_fmt(ppls['pseudo'], 3)} \\\\"
        )
    stats_path = RESULTS / "e4_statistics.json"
    if stats_path.is_file():
        st = json.loads(stats_path.read_text(encoding="utf-8"))
        h1 = st.get("1", {}).get("silver_vs_random", {})
        if h1:
            lines.append(
                r"\midrule"
                + "\n"
                + rf"\multicolumn{{4}}{{@{{}}l@{{}}}}{{\footnotesize Wilcoxon silver vs.\ random (E4b): $p={_fmt(h1.get('p_value'), 4)}$, $d={_fmt(h1.get('cohens_d'), 3)}$.}} \\"
            )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    (TABLES / "e4_oracle.tex").write_text("\n".join(lines), encoding="utf-8")


def write_e4_ctx_table() -> None:
    path = RESULTS / "e4_ctx_dropout_v3.json"
    if not path.is_file():
        (TABLES / "e4_ctx_dropout.tex").write_text(
            "% e4_ctx_dropout.tex — run E4c after forward_3b_ctx_dropout_v3 training\n",
            encoding="utf-8",
        )
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    summary = data.get("summary", {})
    mean_nll = {
        "pseudo": summary.get("mean_nll_pseudo", {}),
        "random": summary.get("mean_nll_random_z", {}),
        "silver": summary.get("mean_nll_silver", {}),
    }
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{E4c: Context-dropout forward replay (50\% Z-only rows during training). Mean perplexity.}",
        r"\label{tab:e4-ctx}",
        r"\small",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"$h$ & Pseudo-$Z$ & Random-$Z$ & Silver-$Z$ \\",
        r"\midrule",
    ]
    max_gap = 0.0
    for h in sorted(int(k) for k in mean_nll["pseudo"].keys()):
        ppls = {}
        for arm, src in [("pseudo", "pseudo"), ("random", "random"), ("silver", "silver")]:
            nll = mean_nll[src].get(str(h))
            ppls[arm] = math.exp(nll) if nll is not None else float("nan")
        max_gap = max(
            max_gap,
            abs(ppls["silver"] - ppls["pseudo"]),
            abs(ppls["silver"] - ppls["random"]),
            abs(ppls["random"] - ppls["pseudo"]),
        )
        lines.append(
            f"{h} & {_fmt(ppls['pseudo'], 3)} & {_fmt(ppls['random'], 3)} & {_fmt(ppls['silver'], 3)} \\\\"
        )
    lines.extend(
        [
            r"\midrule",
            rf"\multicolumn{{4}}{{@{{}}l@{{}}}}{{\footnotesize Max pairwise PPL gap $= {_fmt(max_gap, 3)}$ (intermediate vs.\ E4a, below oracle).}} \\",
        ]
    )
    ctx_stats = RESULTS / "e4_statistics_ctx_v3.json"
    if ctx_stats.is_file():
        st = json.loads(ctx_stats.read_text(encoding="utf-8"))
        h1 = st.get("1", {}).get("silver_vs_random", {})
        if h1:
            lines.append(
                rf"\multicolumn{{4}}{{@{{}}l@{{}}}}{{\footnotesize Wilcoxon silver vs.\ random (E4c): $p={_fmt(h1.get('p_value'), 4)}$, $d={_fmt(h1.get('cohens_d'), 3)}$; not significant at $h\in\{{3,5\}}$.}} \\"
            )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    (TABLES / "e4_ctx_dropout.tex").write_text("\n".join(lines), encoding="utf-8")


def write_fig6_oracle() -> None:
    path = RESULTS / "e4_oracle_v3.json"
    fig_path = REPO / "paper" / "figures" / "fig6_e4_oracle_bars.tex"
    if not path.is_file():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    summary = data.get("summary", {})
    ppls: dict[int, dict[str, float]] = {}
    for h in summary.get("horizons", [1, 3, 5]):
        ppls[int(h)] = {
            "silver": math.exp(summary["mean_nll_silver"][str(h)]),
            "random": math.exp(summary["mean_nll_random_z"][str(h)]),
            "pseudo": math.exp(summary["mean_nll_pseudo"][str(h)]),
        }
    lo = min(v for row in ppls.values() for v in row.values()) - 0.05
    hi = max(v for row in ppls.values() for v in row.values()) + 0.05
    span = max(hi - lo, 0.01)

    def _h(ppl: float) -> float:
        return round(1.5 * (ppl - lo) / span, 2)

    lines = [
        r"\begin{figure}[t]",
        r"  \centering",
        r"  \begin{tikzpicture}[font=\small]",
        r"    \def\barw{0.18}",
    ]
    xpos = {1: 0, 3: 2.2, 5: 4.4}
    for h, x in xpos.items():
        if h not in ppls:
            continue
        row = ppls[h]
        lines.append(
            f"      \\fill[blue!60] ({x},\\barw) rectangle ({x}+\\barw,{_h(row['silver'])});"
        )
        lines.append(
            f"      \\fill[red!60] ({x}+\\barw,\\barw) rectangle ({x}+2*\\barw,{_h(row['random'])});"
        )
        lines.append(
            f"      \\fill[gray!50] ({x}+2*\\barw,\\barw) rectangle ({x}+3*\\barw,{_h(row['pseudo'])});"
        )
        lines.append(f"      \\node at ({x}+1.5*\\barw,-0.3) {{$h={h}$}};")
    lines.extend(
        [
            r"    \draw[->] (-0.2,0) -- (5.8,0);",
            r"    \draw[->] (0,-0.1) -- (0,1.8);",
            r"    \node[left] at (0,0.9) {PPL};",
            r"    \node[blue!60!black] at (4.8,1.6) {\scriptsize Silver};",
            r"    \node[red!60!black] at (4.8,1.45) {\scriptsize Random};",
            r"    \node[gray] at (4.8,1.3) {\scriptsize Pseudo};",
            r"  \end{tikzpicture}",
            r"  \caption{E4b: Z-only oracle forward replay (measured PPL from Table~\ref{tab:e4-oracle}).}",
            r"  \label{fig:e4-oracle}",
            r"\end{figure}",
            "",
        ]
    )
    fig_path.write_text("\n".join(lines), encoding="utf-8")


def write_e5_table() -> None:
    path = RESULTS / "e5_expert_agreement.json"
    if not path.is_file():
        (TABLES / "e5_human.tex").write_text(
            "% e5_human.tex — run scripts/run_expert_dual_rater.py\n",
            encoding="utf-8",
        )
        return
    d = json.loads(path.read_text(encoding="utf-8"))
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Dual API rater agreement (E5) on $n{=}$" + str(d.get("n_dialogues", 25)) + r" stratified test dialogues.}",
        r"\label{tab:e5-human}",
        r"\small",
        r"\begin{tabular}{ll}",
        r"\toprule",
        r"Metric & Value \\",
        r"\midrule",
        f"Rater A model & {d.get('rater_a_model', '---')} \\\\",
        f"Rater B model & {d.get('rater_b_model', '---')} \\\\",
        f"Cohen's $\\kappa$ (KC binary) & {_fmt(d.get('cohen_kappa_kc_binary'), 3)} \\\\",
        f"Silver vs.\ rater A (KC MAE) & {_fmt(d.get('silver_vs_rater_a_kc_mae'), 3)} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    (TABLES / "e5_human.tex").write_text("\n".join(lines), encoding="utf-8")


def write_e3_table() -> None:
    for name in ("mathdial_prefix_curve_inverter.summary.json", "mathdial_prefix_curve.summary.json"):
        path = RESULTS / "identifiability" / name
        if path.is_file():
            break
    else:
        return
    agg = json.loads(path.read_text(encoding="utf-8"))
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Identifiability: mean Bernoulli entropy of predicted mastery vs.\ prefix length $T$.}",
        r"\label{tab:e3-ident}",
        r"\small",
        r"\begin{tabular}{cc}",
        r"\toprule",
        r"Prefix turns $T$ & Mean mastery entropy \\",
        r"\midrule",
    ]
    for key in sorted(agg.keys(), key=lambda k: int(k.replace("mean_entropy_T", "").replace("full", "999"))):
        t = key.replace("mean_entropy_T", "")
        lines.append(f"{t} & {_fmt(agg[key], 4)} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    (TABLES / "e3_identifiability.tex").write_text("\n".join(lines), encoding="utf-8")


def write_related_work_table() -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Positioning of ISS against representative inverse / student models.}",
        r"\label{tab:related}",
        r"\footnotesize",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"System & Dialogue & Joint $(m,C,g)$ & Uncertainty & Identif. & Replay val. \\",
        r"\midrule",
        r"LBM & Answers & Text & Partial & -- & -- \\",
        r"LLMKT & Dialogue & KC & Point & -- & -- \\",
        r"G-R-R & Dialogue & Misc. & Partial & -- & -- \\",
        r"MISTAKE & Answers & Misc. & Partial & Partial & Partial \\",
        r"CSM & Answers & Misc. & -- & -- & Gen. \\",
        r"ADAPT & Behavior & Misc. & -- & -- & Teaching \\",
        r"DKT & Responses & KC & Point & -- & -- \\",
        r"BKT & Responses & KC & Point & -- & -- \\",
        r"\textbf{ISS} & \textbf{Dialogue} & \textbf{Yes} & \textbf{Calib.} & \textbf{E3} & \textbf{E4a--c} \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    (TABLES / "related_work.tex").write_text("\n".join(lines), encoding="utf-8")


def write_setup_table() -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Model and training configuration (ISS v3 overhaul).}",
        r"\label{tab:setup}",
        r"\small",
        r"\begin{tabular}{ll}",
        r"\toprule",
        r"Item & Value \\",
        r"\midrule",
        r"Backbone & Qwen2.5-3B-Instruct \\",
        r"Fine-tuning & QLoRA 4-bit (NF4); $r{=}32$, $\alpha{=}64$ \\",
        r"Silver labels (v3) & 1,429 train dialogues; spread-enforced \\",
        r"Inverter SFT examples & 2,015 \\",
        r"Forward SFT (context) & 11,937 \\",
        r"Forward SFT (oracle Z-only) & 11,937 \\",
        r"Context-dropout mix & 50\% oracle rows \\",
        r"Training epochs & 5 \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    (TABLES / "setup_training.tex").write_text("\n".join(lines), encoding="utf-8")


def write_e7_table() -> None:
    for sub in (RESULTS / "e7_bridge_v3.json", RESULTS / "real" / "e7_run" / "real_experiments.json"):
        if sub.is_file():
            if sub.name.endswith("real_experiments.json"):
                d = json.loads(sub.read_text(encoding="utf-8"))["e7"]
            else:
                d = json.loads(sub.read_text(encoding="utf-8"))
            break
    else:
        return
    if d.get("status") == "skipped":
        lines = [
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{Bridge error-type classification (closed-set prompt, gpt-4o-mini, $n{=}200$).}",
            r"\label{tab:e7-bridge}",
            r"\small",
            r"\begin{tabular}{ll}",
            r"\toprule",
            r"Status & Reason \\",
            r"\midrule",
            r"Not executed & OPENAI\_API\_KEY unavailable in replication run \\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
        (TABLES / "e7_bridge.tex").write_text("\n".join(lines), encoding="utf-8")
        return
    acc = d.get("accuracy_vs_gold", d.get("accuracy", 0))
    n = d.get("n_samples", 0)
    e_topk = d.get("e_topk", 25)
    details = d.get("details", [])
    by_gold: dict[str, list[bool]] = defaultdict(list)
    for row in details:
        gold = str(row.get("gold", ""))
        ok = str(row.get("match", "")).lower() == "true"
        by_gold[gold].append(ok)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{Bridge error-type classification (closed-set prompt, gpt-4o-mini; top-{e_topk} labels from Bridge train).}}",
        r"\label{tab:e7-bridge}",
        r"\small",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Model & $n$ & Accuracy \\",
        r"\midrule",
        f"gpt-4o-mini (ZS) & {n} & {_fmt(acc, 3)} \\\\",
    ]
    if by_gold:
        lines.extend(
            [
                r"\midrule",
                r"\multicolumn{3}{@{}l@{}}{\footnotesize Per gold error type (exact-match accuracy):} \\",
            ]
        )
        for gold in sorted(by_gold.keys(), key=lambda k: (-len(by_gold[k]), k)):
            hits = by_gold[gold]
            lines.append(
                rf"\multicolumn{{3}}{{@{{}}l@{{}}}}{{\footnotesize {gold}: {_fmt(sum(hits)/len(hits), 3)} ($n={len(hits)}$)}} \\"
            )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    (TABLES / "e7_bridge.tex").write_text("\n".join(lines), encoding="utf-8")


def write_aux_table() -> None:
    path = RESULTS / "real" / "20260514T164121Z" / "real_experiments.json"
    if not path.is_file():
        return
    d = json.loads(path.read_text(encoding="utf-8"))
    rows_def = [
        ("e1", "Bridge error type (E)", "train$\\to$val"),
        ("e3", "Bridge error type (E)", "train$\\to$test"),
        ("e4", "Bridge lesson topic", "train$\\to$val"),
        ("e5", "TalkMoves student move", "train$\\to$test"),
        ("e6", "Corpus source (4-way)", "pooled holdout"),
    ]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Auxiliary embedding baselines (MiniLM + logistic regression).}",
        r"\label{tab:aux-bench}",
        r"\small",
        r"\begin{tabular}{llcc}",
        r"\toprule",
        r"Task & Split & Acc. & Macro-F1 \\",
        r"\midrule",
    ]
    for key, task, split in rows_def:
        m = d[key]["metrics"]
        lines.append(
            f"{task} & {split} & {_fmt(m['accuracy'], 3)} & {_fmt(m['macro_f1'], 3)} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    (TABLES / "aux_benchmarks.tex").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    write_e1_table()
    write_e4_table()
    write_e4_oracle_table()
    write_e4_ctx_table()
    write_fig6_oracle()
    write_e5_table()
    write_e3_table()
    write_e7_table()
    write_related_work_table()
    write_aux_table()
    write_setup_table()
    print(f"Wrote tables to {TABLES}")


if __name__ == "__main__":
    main()
