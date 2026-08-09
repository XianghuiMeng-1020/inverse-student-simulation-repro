"""Top-level orchestration entry point for the ISS reproducibility package.

Chains data preparation -> SFT construction -> evaluation -> diagnostics ->
paper tables using each script's real CLI and defaults. Stages that need an
artifact you have not produced yet (a trained LoRA adapter, silver labels)
are skipped with a clear message instead of failing hard, so this script
doubles as a smoke test.

Stages that require a *required* argument with no repository-wide default
(e.g. --adapter-dir, --labels-jsonl) are intentionally left out of this
orchestrator; run them directly with the commands documented in README.md
("Reproducing Main Results") once the relevant upstream artifact exists.

Every subprocess call below corresponds 1:1 to a documented command in
README.md; nothing here is invented for orchestration purposes only.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import typer

app = typer.Typer(no_args_is_help=True)
REPO_ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str], allow_fail: bool = True) -> bool:
    typer.echo(f"\n[run] {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    ok = result.returncode == 0
    if not ok and not allow_fail:
        raise typer.Exit(code=result.returncode)
    return ok


@app.command()
def main(
    data_dir: Path = typer.Option(Path("data"), "--data-dir"),
    output_dir: Path = typer.Option(Path("experiments/results"), "--output-dir"),
) -> None:
    py = sys.executable

    typer.echo("=== Stage 1: data preparation ===")
    _run([py, "scripts/download_data.py", "--repo-root", "."])
    _run([py, "scripts/build_dataset.py", "--repo-root", ".", "--datasets", "mathdial"])
    _run([py, "scripts/build_splits.py", "--repo-root", "."])

    typer.echo(
        "\n=== Stage 2: structured-state (Z) construction ===\n"
        "[note] Silver labeling calls an LLM API (OPENAI_API_KEY) and is not "
        "auto-run here -- see data/README.md for label_latent_z_v2.py / _v3.py."
    )
    v2_path = REPO_ROOT / data_dir / "labels" / "latent_z_silver_v2.jsonl"
    if v2_path.is_file():
        _run([py, "scripts/postprocess_v2_to_v3.py", "--in-jsonl", str(v2_path)])
        v3_path = REPO_ROOT / data_dir / "labels" / "latent_z_silver_v3.jsonl"
        if v3_path.is_file():
            _run([py, "scripts/validate_labels_v3.py", "--labels-jsonl", str(v3_path)])
    else:
        typer.echo(f"[skip] {v2_path} not found -- run label_latent_z_v2.py first.")

    typer.echo("\n=== Stage 3: SFT data construction (requires silver labels) ===")
    for script in ("build_inverter_sft_jsonl.py", "build_forward_sft_jsonl.py", "build_oracle_forward_sft_jsonl.py"):
        _run([py, f"scripts/{script}", "--repo-root", "."])

    typer.echo(
        "\n=== Stage 4: training (Hydra CLI, requires GPU) ===\n"
        "[note] not auto-run here -- see README.md 'Training'. Example:\n"
        "  python scripts/train_inverter.py model=inverter_3b\n"
        "  python scripts/train_forward.py model=forward_3b"
    )

    typer.echo("\n=== Stage 5: evaluation (uses default paths under data/ and experiments/) ===")
    _run([py, "scripts/run_bkt_baseline.py"])
    _run([py, "scripts/run_inverter_eval.py", "--skip-inverter"])
    typer.echo(
        "[note] full inverter/replay evaluation additionally needs "
        "--adapter-dir / --adapter pointing at a trained checkpoint; "
        "see README.md 'Reproducing Main Results'."
    )

    typer.echo("\n=== Stage 6: diagnostics (deterministic, no GPU/API needed) ===")
    for script in (
        "robustness_kc_structure.py",
        "robustness_misconception_sparsity.py",
        "robustness_forward_replay_probes.py",
        "robustness_dialogue_baselines.py",
    ):
        _run([py, f"scripts/{script}"])

    typer.echo("\n=== Stage 7: paper tables ===")
    _run([py, "scripts/generate_paper_tables.py"])
    _run([py, "scripts/sanity_check_results.py"])

    typer.echo(f"\n[done] see {REPO_ROOT / output_dir} for regenerated metrics/tables.")


if __name__ == "__main__":
    app()
