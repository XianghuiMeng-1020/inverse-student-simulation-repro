"""Eedi NeurIPS 2020 auxiliary loader (plan p4-11 scaffold).

The competition drop is distributed via Kaggle / NeurIPS archives with
multiple mirrors.  We intentionally **do not** hard-pin a Hugging Face dataset id
here (IDs drift); ingest a local CSV/Parquet export instead.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def load_eedi_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        msg = f"Eedi export not found at {path}"
        raise FileNotFoundError(msg)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def row_to_record(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "question_id": str(row.get("QuestionId", row.get("question_id", ""))),
        "misconception_id": str(row.get("MisconceptionId", row.get("misconception_id", ""))),
    }
