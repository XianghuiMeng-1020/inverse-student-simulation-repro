<div align="center">

# Inverse Student Simulation
### Reproducibility Package for Structured Student-State Validation

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6%2B-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-transformers%20%7C%20datasets-yellow)](https://huggingface.co/)
[![Reproducibility](https://img.shields.io/badge/reproducibility-package-brightgreen)](#-reproducing-main-results)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

<br/>

*Given a dialogue between a tutor and a student, can we recover a
structured description of what the student knows, what they misunderstand,
and how they regulate their own learning — and does that recovered state
actually change how a model predicts what the student says next?*

This repository is the reproducibility package for **Inverse Student
Simulation (ISS)**: a framework that inverts tutor–student dialogue into a
structured latent state `Z = (m, C, g)` — knowledge-component (KC) mastery,
active misconceptions, and exploratory metacognitive indicators — and then
*audits* that inversion through calibration checks, identifiability
analysis, and counterfactual forward replay.

</div>

---

## Overview

Most published student-simulation work runs **forward**: given a latent
state, generate plausible student behavior. Practical use in intelligent
tutoring systems (ITS) also needs the **inverse** direction: given an
observed dialogue, recover a structured, auditable state.

This repository packages the full computational pipeline behind that
question:

1. **Does structured state vary meaningfully across dialogues**, or does an
   evidence-free heuristic explain most of the apparent structure?
2. **Can the state be recovered from dialogue evidence** beyond what a
   simple label-prior or classical baseline (BKT, GPT-4o zero-shot) already
   captures?
3. **Does the recovered state actually influence downstream next-response
   prediction**, or does rich dialogue context make it redundant?

The honest, audited answer — reported in the accompanying manuscript — is
**uneven**: KC mastery shows partial, positive intra-dialogue structure but
near-chance cross-dialogue discrimination; misconception ranking metrics
are inflated by label-set uniformity; metacognitive scalars are exploratory
and largely unrecovered; and a context-conditioned forward model mostly
bypasses the structured state unless dialogue context is deliberately
restricted. This package lets you regenerate every one of those findings
from public data.

---

## ✨ Highlights

- **Dialogue-to-state inversion** — a QLoRA-finetuned `Qwen/Qwen2.5-3B-Instruct`
  inverter that maps a dialogue prefix to a structured `LatentZ` (30 KC
  mastery probabilities, 70 misconception probabilities, 4 metacognitive
  scalars), with deterministic JSON-schema repair for near-miss model output.
- **Structured-state auditing** — calibration (Brier, ECE), within-dialogue
  vs. pooled cross-dialogue discrimination (Spearman, AUC), and a
  misconception label-sparsity audit that distinguishes genuine
  discrimination from label-prior artifacts.
- **Simple-baseline and non-degeneracy checks** — a from-scratch
  multi-skill BKT baseline, a zero-shot GPT-4o baseline, and an
  evidence-free Pseudo-Z heuristic used as label-prior controls throughout.
- **Counterfactual forward replay** — a context-conditioned forward
  simulator and a Z-only ("oracle") forward simulator trained under an
  information bottleneck, evaluated with Silver-Z / Pseudo-Z / Random-Z
  substitution, paired Wilcoxon tests, and effect sizes.
- **Reproducible statistics** — dialogue-level aggregation, paired
  significance testing, and bootstrap confidence intervals throughout.
- **Two ways in**: a five-minute, dependency-light [`tutorial/`](tutorial/)
  that exercises the real schema/metric code on toy data with no GPU or API
  key, and the full [`scripts/`](scripts/) pipeline for the actual MathDial
  experiments.

---

## 📦 Repository Structure

```
.
├── README.md                 <- you are here
├── LICENSE                    MIT (code); see below for data/model licenses
├── requirements.txt            minimal pinned dependencies
├── .env.example                 template for API keys (copy to .env)
│
├── configs/                   Hydra configuration (model, training, eval)
│   ├── config.yaml              root config (paths, W&B, seeds)
│   ├── data/                    per-corpus config (mathdial, bridge, cima, talkmoves, ...)
│   ├── model/                   inverter_3b.yaml, forward_3b.yaml (QLoRA hyperparameters)
│   ├── train/                   lora_default.yaml (epochs, LR, batch size, seeds)
│   └── eval/                    default.yaml (metrics, replay horizons, bootstrap, tests)
│
├── src/iss/                   core library (imported by every script below)
│   ├── schema/                   LatentZ pydantic schema, KC/misconception ontologies,
│   │                             JSON-schema grammar + deterministic repair
│   ├── data/                     per-corpus loaders -> unified Dialogue/turn schema,
│   │                             split-manifest construction, silver-label I/O
│   ├── forward/                  Pseudo-Z / Random-Z construction, SFT row builders
│   │                             (full-context, oracle, context-dropout), ForwardSimulator
│   ├── inverter/                  JointInverter model, SFT row builder, prompts, losses
│   ├── training/                  shared causal-LM QLoRA SFT loop
│   ├── eval/                      metrics (Brier/ECE/AUC/entropy), bootstrap CIs,
│   │                             counterfactual replay (NLL/perplexity), identifiability
│   ├── baselines/                 from-scratch multi-skill BKT, GPT-4o zero-shot,
│   │                             OpenRouter client
│   └── experiments/               dialogue-text utilities, E7 Bridge probe, embedding
│                                 baselines (sentence-transformers)
│
├── scripts/                   full reproduction pipeline (35 CLI entry points)
│   ├── reproduce_results.py       top-level orchestrator (chains the stages below)
│   ├── download_data.py, build_dataset.py, build_splits.py       <- data prep
│   ├── label_latent_z_v2.py, label_latent_z_v3.py,
│   │   postprocess_v2_to_v3.py, validate_labels_v3.py            <- structured-state (Z) construction
│   ├── build_inverter_sft_jsonl.py, build_forward_sft_jsonl.py,
│   │   build_oracle_forward_sft_jsonl.py,
│   │   build_context_dropout_forward_sft_jsonl.py                <- SFT data construction
│   ├── train_inverter.py, train_forward.py                       <- QLoRA training
│   ├── run_inverter_eval.py, run_bkt_baseline.py,
│   │   run_identifiability_inverter.py                           <- E1 / E3 evaluation
│   ├── run_replay_eval.py, run_replay_statistics.py               <- E4 counterfactual replay
│   ├── compute_label_agreement_e5.py, run_expert_dual_rater.py    <- E5 label agreement
│   ├── run_e7_v3.py                                               <- E7 Bridge zero-shot probe
│   ├── robustness_kc_structure.py,
│   │   robustness_misconception_sparsity.py,
│   │   robustness_forward_replay_probes.py,
│   │   robustness_dialogue_baselines.py                          <- diagnostics / audits
│   └── generate_paper_tables.py, sanity_check_results.py          <- table generation, sanity checks
│
├── tutorial/                   5-minute, no-GPU/no-API walkthrough (see tutorial/README.md)
│   ├── 01_inspect_schema.py, 02_construct_states.py,
│   │   03_evaluate_kc_recovery.py, 04_misconception_ranking.py
│   └── toy_dialogues.json         3 synthetic tutoring dialogues (not MathDial)
│
├── tests/                      pytest unit tests for src/iss (schema, metrics, splits, ...)
│
├── data/                        README.md only — see "Data Preparation" below
│   └── README.md                 dataset sources, prep commands, expected layout
│
└── results/
    ├── expected_metrics.json      verification targets (final manuscript numbers)
    └── README.md
```

Everything under `data/`, `experiments/`, and `tutorial/outputs/` other than
the checked-in `README.md`/`expected_metrics.json` files is generated by
the scripts and is gitignored — nothing large is committed to this
repository.

---

## ⚙️ Installation

**Requirements:** Python 3.11 or 3.12; a CUDA GPU is required only for
training/inference stages (QLoRA fine-tuning and replay evaluation of
`Qwen2.5-3B-Instruct`). Everything in `tutorial/`, the data-preparation
stages, and the deterministic diagnostic scripts run on CPU.

```bash
git clone https://github.com/<your-account>/inverse-student-simulation-repro.git
cd inverse-student-simulation-repro

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

Then copy the environment template and fill in only the keys you need
(silver labeling and the E5/E7 API baselines need `OPENAI_API_KEY`;
everything else works without any key):

```bash
cp .env.example .env
```

Verify the install with the test suite and the schema tutorial step (both
run in a few seconds, no GPU/API required):

```bash
python -m pytest tests -q
python tutorial/01_inspect_schema.py
```

---

## 📚 Data Preparation

Full details, licenses, and expected row/schema layout are in
[`data/README.md`](data/README.md). Short version:

```bash
# 1. Fetch raw corpora (MathDial via Hugging Face Hub; CIMA/TalkMoves via shallow git clone)
python scripts/download_data.py --repo-root .

# 2. Build unified parquet shards: data/processed/<name>/{train,dev,test}.parquet
python scripts/build_dataset.py --repo-root . --datasets mathdial

# 3. Deterministic 70/10/20 split manifests: data/splits/<name>.json
python scripts/build_splits.py --repo-root .
```

The **primary** dataset is [MathDial](https://huggingface.co/datasets/eth-nlped/mathdial)
(CC-BY-4.0, English math tutoring dialogues); all primary inversion,
identifiability, replay, and label-agreement results are computed on its
394-dialogue test split. Bridge, TalkMoves, and CIMA are used only for the
auxiliary probes described below (E7 zero-shot error-type classification
and embedding-baseline sanity checks) — they do **not** ground the primary
ISS results.

### Structured-state (`Z`) construction

Silver `LatentZ` labels are produced by prompting an LLM (the manuscript
uses `gpt-4o-mini`) and are **not redistributed** in this repository
because generating them requires your own API key:

```bash
# v2 labeling with spread-enforcement constraints (requires OPENAI_API_KEY)
python scripts/label_latent_z_v2.py --help

# deterministic v2 -> v3 post-processing (no API call; template-audit + normalization)
python scripts/postprocess_v2_to_v3.py --in-jsonl data/labels/latent_z_silver_v2.jsonl

# sanity-check the resulting v3 label set (corpus KC std, sparsity, schema validity)
python scripts/validate_labels_v3.py --labels-jsonl data/labels/latent_z_silver_v3.jsonl
```

If you do not have an API key, you can still exercise the exact same
label-construction *logic* deterministically and offline via
[`tutorial/02_construct_states.py`](tutorial/02_construct_states.py) (Pseudo-Z /
Random-Z construction, no LLM call).

---

## 🚀 Quick Start

If you just want to see the pipeline work end-to-end without a GPU or API
key, start with the tutorial:

```bash
cd tutorial
python 01_inspect_schema.py
python 02_construct_states.py
python 03_evaluate_kc_recovery.py
python 04_misconception_ranking.py
cd ..
```

See [`tutorial/README.md`](tutorial/README.md) for what each step
demonstrates and how it maps to the full-scale scripts.

For the real pipeline, the top-level orchestrator chains data preparation,
SFT construction, evaluation, and diagnostics using each script's real CLI
(training and full replay evaluation require a GPU and are described
separately, since they are the expensive stages):

```bash
python scripts/reproduce_results.py
```

Stages that need an artifact you have not produced yet (a trained LoRA
adapter, silver labels) print a clear `[skip]`/`[note]` message rather than
failing, so this same command doubles as an installation smoke test.

---

## 📊 Reproducing Main Results

Each result family in the manuscript maps to a specific command below. All
commands assume `data/processed/mathdial/{train,dev,test}.parquet` and (for
anything past labeling) `data/labels/latent_z_silver_v3.jsonl` already
exist (see "Data Preparation").

### 1. Silver supervision and SFT data

```bash
python scripts/label_latent_z_v2.py --help                 # LLM silver labeling (API key)
python scripts/postprocess_v2_to_v3.py --in-jsonl <v2.jsonl>  # deterministic V2 -> V3
python scripts/validate_labels_v3.py --labels-jsonl <v3.jsonl>

python scripts/build_inverter_sft_jsonl.py --repo-root .
python scripts/build_forward_sft_jsonl.py --repo-root .
python scripts/build_oracle_forward_sft_jsonl.py --repo-root . --labels-jsonl <v3.jsonl>
python scripts/build_context_dropout_forward_sft_jsonl.py --labels-jsonl <v3.jsonl>
```

### 2. Training (QLoRA, requires GPU)

Training uses Hydra config composition (`configs/model/*.yaml`,
`configs/train/lora_default.yaml`) rather than argparse flags:

```bash
# Joint inverter: Qwen2.5-3B-Instruct + QLoRA (r=32, alpha=64, 4-bit NF4), 5 epochs
python scripts/train_inverter.py model=inverter_3b

# Context-conditioned forward simulator: QLoRA (r=16, alpha=32), full dialogue history + Z
python scripts/train_forward.py model=forward_3b
```

Repeat `train_inverter.py` with `project.seed=1` and `project.seed=2` to
reproduce the three-seed inverter variability reported in the manuscript.
Checkpoints are written to `experiments/checkpoints/<run_name>/`.

### 3. KC mastery: inversion accuracy and baseline comparison (E1)

```bash
# Fine-tuned inverter (Brier, pooled AUC, F1@5, MRR@10 at prefix lengths T={2,4,8,full})
python scripts/run_inverter_eval.py --adapter-dir experiments/checkpoints/<run>/inverter_lora

# Classical multi-skill BKT reference
python scripts/run_bkt_baseline.py

# Evidence-budget / identifiability curve (Bernoulli entropy vs. prefix length T)
python scripts/run_identifiability_inverter.py --adapter-dir experiments/checkpoints/<run>/inverter_lora
```

### 4. KC structure: within-dialogue recoverability (extended E1)

```bash
# Within-dialogue Spearman, top-3 KC overlap, pooled-vs-within-dialogue AUC contrast
python scripts/robustness_kc_structure.py
```

### 5. Misconception label analysis and sparsity audit

```bash
# Active-label prevalence, label-prior baseline, F1@5/MRR sensitivity to label uniformity
python scripts/robustness_misconception_sparsity.py
```

### 6. Forward replay: counterfactual Z-sensitivity (E4a-c)

```bash
# E4a: context-conditioned model, Silver-Z / Pseudo-Z / Random-Z substitution, horizons 1/3/5
python scripts/run_replay_eval.py --repo-root . --e4-grid \
    --adapter experiments/checkpoints/<run>/forward_lora \
    --parquet data/processed/mathdial/test.parquet \
    --silver-labels data/labels/latent_z_silver_v3.jsonl

# E4b: Z-only oracle forward model (train with build_oracle_forward_sft_jsonl.py first)
python scripts/run_replay_eval.py --repo-root . --e4-grid \
    --adapter experiments/checkpoints/<oracle_run>/forward_lora \
    --silver-labels data/labels/latent_z_silver_v3.jsonl

# E4c: context-dropout forward model (train with build_context_dropout_forward_sft_jsonl.py first)
python scripts/run_replay_eval.py --repo-root . --e4-grid \
    --adapter experiments/checkpoints/<dropout_run>/forward_lora \
    --silver-labels data/labels/latent_z_silver_v3.jsonl

# Paired Wilcoxon signed-rank tests, Holm-style correction, effect sizes, dialogue-clustered CIs
python scripts/run_replay_statistics.py --replay-json experiments/results/<e4_output>.json
```

### 7. Label agreement (E5) and Bridge zero-shot probe (E7)

```bash
python scripts/compute_label_agreement_e5.py             # silver v3 vs. v2 agreement (no API)
python scripts/run_expert_dual_rater.py --n 25            # dual API-rater agreement (needs API key)
python scripts/run_e7_v3.py --n-samples 200                # Bridge closed-set error-type probe
```

### 8. Assemble manuscript tables

```bash
python scripts/generate_paper_tables.py     # writes paper/tables/*.tex from experiments/results/*.json
python scripts/sanity_check_results.py      # cross-checks generated JSON against expected structure
```

---

## 🧪 Diagnostic Analyses

These scripts are deterministic (no GPU, no API key) once their upstream
JSON inputs exist, and correspond to the "audit" framing of the
manuscript rather than headline numbers:

| Script | Diagnoses |
|---|---|
| `robustness_kc_structure.py` | Disentangles near-chance pooled cross-dialogue AUC from positive within-dialogue Spearman correlation. |
| `robustness_misconception_sparsity.py` | Quantifies active-label prevalence and the label-prior baseline that near-perfect F1/MRR must be compared against. |
| `robustness_forward_replay_probes.py` | Cross-checks E4 replay deltas for consistency across horizons and Z-conditions. |
| `robustness_dialogue_baselines.py` | Simple non-LLM dialogue-level baselines used as sanity floors/ceilings. |
| `run_e1_multiseed_aggregate.py` | Aggregates the three inverter seeds (42, 1, 2) into mean ± std. |
| `sanity_check_results.py` | Structural completeness check over all `experiments/results/*.json` files. |

---

## ✅ Expected Outputs

[`results/expected_metrics.json`](results/expected_metrics.json) lists the
final reported checkpoints (KC Brier/AUC/ECE/Spearman, misconception
F1@5/MRR vs. the label-prior baseline, E4a-c replay deltas with Wilcoxon
p-values and effect sizes, E5 agreement, E7 accuracy). These are
verification targets only — no script hard-codes them; regenerate and
diff against this file. Because silver labeling depends on a specific LLM
snapshot and training is stochastic across seeds, expect small
(sub-percent to low-single-digit) deviations rather than bit-exact
reproduction; large qualitative disagreements (e.g. pooled AUC moving from
"near chance" to "clearly discriminative") would indicate a real
divergence worth investigating.

---

## 🖥 Computational Requirements

| Stage | Hardware | Notes |
|---|---|---|
| Tutorial (`tutorial/`) | CPU, any modern laptop | Seconds; no GPU/API. |
| Data prep, splits, diagnostics | CPU | Minutes. |
| Silver labeling (`label_latent_z_v2/_v3.py`) | CPU + API key | Cost/time scale with corpus size and API rate limits. |
| QLoRA training (`train_inverter.py`, `train_forward.py`) | 1x GPU, >=16 GB VRAM (4-bit NF4 QLoRA) | Qwen2.5-3B-Instruct, 5 epochs; the reported runs used a single 32 GB consumer GPU. |
| Inverter/replay evaluation | 1x GPU (or CPU with `--max-gpu-gb 0` and patience) | Batched generation; `run_replay_eval.py --e4-grid` over the full 394-dialogue test set is the most expensive evaluation stage. |

`Qwen/Qwen2.5-3B-Instruct` downloads automatically from the Hugging Face
Hub on first use (Apache-2.0 licensed model card).

---

## 🛠 Troubleshooting

<details>
<summary>ImportError: No module named 'iss'</summary>

Run scripts from the repository root (they insert `src/` onto `sys.path`
relative to their own file location) or `pip install -e .` is not required
for the scripts themselves — but if you are writing your own code against
`src/iss`, add `src/` to `PYTHONPATH`.
</details>

<details>
<summary>Scripts that reference `--adapter-dir` / `--adapter` fail with "no such file"</summary>

Those flags must point at a LoRA adapter directory produced by
`train_inverter.py` / `train_forward.py` (default location:
`experiments/checkpoints/<run_name>/{inverter,forward}_lora`). Run the
training stage first, or pass `--skip-inverter` to `run_inverter_eval.py`
to evaluate only the Pseudo-Z / BKT baselines without a trained checkpoint.
</details>

<details>
<summary>Diagnostic scripts (robustness_*.py) raise FileNotFoundError</summary>

These scripts read `experiments/results/*.json` and `data/labels/*.jsonl`
produced by earlier pipeline stages (labeling, inverter evaluation). This
is expected on a fresh clone before you have run those stages — see
"Reproducing Main Results" for the required order.
</details>

<details>
<summary>bitsandbytes / 4-bit quantization fails to load on your platform</summary>

4-bit QLoRA (`bitsandbytes`) requires a CUDA GPU and a supported Linux/Windows
build. On unsupported platforms, set `load_in_4bit: false` in
`configs/model/inverter_3b.yaml` / `forward_3b.yaml` and reduce batch size
if full-precision does not fit in memory.
</details>

<details>
<summary>Rate limits / cost during silver labeling</summary>

`label_latent_z_v2.py` / `_v3.py` use a thread pool with configurable sleep
between requests (`--sleep-s`). Reduce `--limit` for a small pilot run
before labeling the full corpus.
</details>

---

## ⚖️ License

Source code in this repository is released under the [MIT License](LICENSE).
Third-party datasets and models retain their own licenses — see
[`data/README.md`](data/README.md) and the notice at the bottom of
[`LICENSE`](LICENSE) (MathDial: CC-BY-4.0; Bridge: see HF dataset card;
CIMA: CC BY-NC-SA 2.5; TalkMoves: CC BY-NC-SA 4.0; Qwen2.5-3B-Instruct:
Apache-2.0).

---

<div align="center">
<sub>Structured student-state validation via inverse student simulation.</sub>
</div>
