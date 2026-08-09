"""ISS command-line interface — thin Typer wrapper around scripts.

Exposed as `iss` after `uv pip install -e .`.
"""

from __future__ import annotations

import typer
from rich.console import Console

from iss import __version__

app = typer.Typer(
    name="iss",
    help="Inverse Student Simulation — ICCE 2026 C1 submission.",
    no_args_is_help=True,
)
console = Console()


@app.command()
def version() -> None:
    """Print the installed package version."""
    console.print(f"[bold cyan]iss[/bold cyan] version [green]{__version__}[/green]")


@app.command()
def doctor() -> None:
    """Verify environment: python, torch, CUDA, key deps."""
    import platform
    import sys

    console.rule("[bold]ISS environment doctor[/bold]")
    console.print(f"Python    : {sys.version.split()[0]} ({platform.platform()})")

    try:
        import torch

        console.print(f"PyTorch   : {torch.__version__}")
        console.print(f"CUDA avail: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            console.print(f"CUDA ver  : {torch.version.cuda}")
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
                cap = torch.cuda.get_device_capability(i)
                console.print(
                    f"GPU {i}     : {name} ({mem:.1f} GiB, sm_{cap[0]}{cap[1]})"
                )
    except ImportError:
        console.print("[yellow]PyTorch not installed[/yellow]")

    try:
        import transformers

        console.print(f"HF trans  : {transformers.__version__}")
    except ImportError:
        console.print("[yellow]transformers not installed[/yellow]")

    try:
        import peft

        console.print(f"peft      : {peft.__version__}")
    except ImportError:
        console.print("[yellow]peft not installed[/yellow]")

    console.rule()


if __name__ == "__main__":
    app()
