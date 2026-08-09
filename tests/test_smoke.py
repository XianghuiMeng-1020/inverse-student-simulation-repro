"""Smoke tests for repo bootstrap."""

from __future__ import annotations

import pytest


def test_package_importable() -> None:
    import iss

    assert iss.__version__ == "0.1.0"


def test_cli_module_loads() -> None:
    from iss.cli import app

    assert app is not None


def test_logging_module_loads() -> None:
    from iss.logging import init_env, init_logger, init_wandb

    init_env(".env")
    init_logger(level="DEBUG")
    assert callable(init_wandb)


@pytest.mark.parametrize("subpkg", ["data", "schema", "forward", "inverter", "baselines", "eval", "analysis"])
def test_subpackages_importable(subpkg: str) -> None:
    __import__(f"iss.{subpkg}")
