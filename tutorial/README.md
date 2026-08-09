# 🎓 Tutorial: Understanding the ISS Pipeline in 5 Minutes

This is a **self-contained, no-GPU, no-API-key** walkthrough of the core
ideas in the reproducibility package, aimed at readers who want to
understand *how the pieces fit together* before running the full
pipeline in `../scripts/`. Everything here runs in a few seconds on a
laptop CPU using the real `src/iss` library code (not a re-implementation),
applied to three tiny synthetic dialogues instead of MathDial.

**What this tutorial is not**: it does not reproduce the paper's reported
numbers (those require the real MathDial corpus, silver LLM labels, and a
fine-tuned Qwen2.5-3B inverter/forward simulator — see the top-level
`README.md`, "Reproducing Main Results"). It exists purely to make the
structured-state schema, the baselines, and the evaluation metrics
concrete and inspectable.

## Steps

| Script | What it shows |
|---|---|
| `01_inspect_schema.py` | The fixed ontology (30 KCs, 70 misconception codes, 4 metacognitive scalars) and the deterministic JSON-repair path that coerces near-miss LLM output into a valid `LatentZ`. |
| `02_construct_states.py` | Builds **Pseudo-Z** (evidence-free lexical heuristic) and **Random-Z** (IID Uniform control) for 3 toy tutoring dialogues, using the exact same functions (`iss.forward.pseudo_z`) used at full scale. |
| `03_evaluate_kc_recovery.py` | Computes KC Brier score, ECE, within-dialogue Spearman correlation, and top-3 mastery overlap — the same metric definitions behind the paper's E1/KC-structure tables. |
| `04_misconception_ranking.py` | Reproduces, mechanically, the paper's central misconception caveat: a "label-prior" baseline that ignores the dialogue entirely can match a dialogue-aware prediction on F1@5/MRR when the gold active set is uniform across dialogues. |

## Run it

```bash
cd tutorial
python 01_inspect_schema.py
python 02_construct_states.py
python 03_evaluate_kc_recovery.py
python 04_misconception_ranking.py
```

Only `numpy`, `scipy`, `scikit-learn`, and `pydantic` are required (a
subset of `requirements.txt`); no dataset download, no LLM API call, no
GPU. Outputs are written to `tutorial/outputs/` (gitignored).

## From toy to real

Each tutorial step has a direct full-scale counterpart:

| Tutorial step | Full-scale equivalent |
|---|---|
| `01_inspect_schema.py` | `src/iss/schema/*` (used by every script) |
| `02_construct_states.py` | `scripts/label_latent_z_v2.py` + `scripts/postprocess_v2_to_v3.py` (real silver labels) |
| `03_evaluate_kc_recovery.py` | `scripts/run_inverter_eval.py`, `scripts/robustness_kc_structure.py` |
| `04_misconception_ranking.py` | `scripts/robustness_misconception_sparsity.py` |

Once you understand this tutorial, proceed to the top-level `README.md`
for the full data-preparation, training, evaluation, and diagnostic
pipeline on the public MathDial corpus.
