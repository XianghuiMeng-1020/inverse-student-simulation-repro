# Data

No raw or processed data is committed to this repository. All corpora are
public and are fetched/derived by the scripts below into the (gitignored)
subdirectories listed.

## Primary dataset

| | |
|---|---|
| **Name** | MathDial |
| **Source** | Hugging Face Hub, [`eth-nlped/mathdial`](https://huggingface.co/datasets/eth-nlped/mathdial) |
| **License** | CC-BY-4.0 |
| **Role** | Primary English math-tutoring corpus. All primary inversion (E1), identifiability (E3), replay (E4), and label-agreement (E5) results are computed on the MathDial test split (394 dialogues). |
| **Expected volume** | 2,861 dialogues total; silver v3 labels cover 1,429 train + 394 test dialogues used for training/evaluation. |

MathDial downloads automatically on first `datasets.load_dataset("eth-nlped/mathdial")`
call inside `scripts/build_dataset.py` — no manual step required beyond
having network access (and, if the dataset requires it, `HF_TOKEN` set in `.env`).

## Auxiliary resources

These are only used for the auxiliary probes described in the manuscript
(Bridge zero-shot error-type probe, embedding baselines). They do **not**
ground the primary ISS inversion/replay results.

| Dataset | Source | License | Role |
|---|---|---|---|
| **Bridge** | HF Hub, [`rose-e-wang/bridge`](https://huggingface.co/datasets/rose-e-wang/bridge) | see dataset card | Zero-shot closed-set error-type probe (E7); not used for inverter training. |
| **TalkMoves** | GitHub, [`SumnerLab/TalkMoves`](https://github.com/SumnerLab/TalkMoves) | CC BY-NC-SA 4.0 | Auxiliary discourse-structure benchmark only. |
| **CIMA** | GitHub, [`kstats/CIMA`](https://github.com/kstats/CIMA) | CC BY-NC-SA 2.5 | Corpus-source classification sanity check only. |

## Preparation pipeline

```bash
# 1. Fetch raw corpora (HF auto-download + shallow git clones)
python scripts/download_data.py --repo-root .

# 2. Build unified per-dataset parquet shards -> data/processed/<name>/{train,dev,test}.parquet
python scripts/build_dataset.py --repo-root . --datasets mathdial,bridge,cima,talkmoves

# 3. Deterministic 70/10/20 split manifests -> data/splits/<name>.json
python scripts/build_splits.py --repo-root .
```

## Expected directory layout after preparation

```
data/
├── raw/            # git-cloned CIMA / TalkMoves (gitignored)
├── processed/       # unified parquet shards (gitignored)
│   └── mathdial/{train,dev,test}.parquet
├── splits/         # split manifests (gitignored)
├── labels/         # silver LatentZ JSONL labels (gitignored; see below)
├── forward_sft/    # forward-simulator SFT JSONL (gitignored)
└── inverter_sft/   # inverter SFT JSONL (gitignored)
```

## Unified row schema

Every processed row follows the same shape (see `src/iss/schema/latent.py`):
`dialogue_id`, a `turns_json` list of `{speaker, text}` turns, and a free-form
`metadata_json` (e.g. the source math problem for MathDial).

## Silver labels (structured state `Z`)

Silver `LatentZ` labels are LLM-annotated JSONL files
(`data/labels/latent_z_*.jsonl`), produced by the labeling scripts and
consumed by SFT construction / evaluation:

```bash
# v1/base labeling (OpenAI/OpenRouter; requires OPENAI_API_KEY or OPENROUTER_API_KEY)
python scripts/label_latent_z_openai.py --help

# v2 labeling with spread-enforcement constraints
python scripts/label_latent_z_v2.py --help

# deterministic v2 -> v3 post-processing (no API calls)
python scripts/postprocess_v2_to_v3.py --help
```

Because these steps call a paid LLM API, we do not redistribute the raw
labels; document the exact model/version used in your run (the manuscript
uses `gpt-4o-mini` for v2/v3 labeling — see `scripts/label_latent_z_v3.py`).

## Human annotation tables

If you regenerate any inter-rater agreement tables, use generic labels
(`coder_a`, `coder_b`) for anonymous raters rather than real names unless a
specific coder identity is scientifically necessary. `scripts/run_expert_dual_rater.py`
already labels its two API raters generically (`rater_a_model`, `rater_b_model`).
