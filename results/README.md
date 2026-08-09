# Results

This directory holds verification targets, not authoritative outputs.

- **`expected_metrics.json`** — the final numerical checkpoints reported in the
  manuscript (MathDial test set, mostly `n=394`). Use these to sanity-check a
  fresh run: after running the evaluation/diagnostic scripts, compare your
  generated `experiments/results/*.json` against the numbers here.
- Nothing in this file is consumed by the reproduction scripts themselves —
  it exists purely so a reviewer can tell "did my run land in the right
  ballpark" without re-reading the paper.

Regenerated run outputs (from `scripts/run_inverter_eval.py`,
`scripts/run_replay_eval.py`, `scripts/generate_paper_tables.py`, etc.) are
written to `experiments/results/` at the repository root, which is
gitignored. See the top-level `README.md`, section "Reproducing Main
Results", for the exact commands that populate each metric family.
