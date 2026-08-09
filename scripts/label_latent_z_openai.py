"""LLM-assisted ``LatentZ`` labeling — parallel, resume-safe (plan p5-03 / p7-06).

Supports OpenAI **and** OpenRouter (pass ``--base-url https://openrouter.ai/api/v1``).
Uses ``--workers`` threads to call the API concurrently for speed.
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path

import openai
import pandas as pd
import typer
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from iss.inverter.prompts import build_inverter_messages
from iss.schema.grammar import validate_latent_z_json
from iss.schema.latent import DialogueTurn
from iss.schema.repair import repair_latent_z_json

app = typer.Typer(no_args_is_help=True)


def _load_done_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    done: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            done.add(str(rec["dialogue_id"]))
        except (json.JSONDecodeError, KeyError):
            continue
    return done


def _iter_parquet_rows(repo_root: Path, parquet_paths: list[Path]) -> list[tuple[str, list[DialogueTurn]]]:
    rows: list[tuple[str, list[DialogueTurn]]] = []
    for pq in parquet_paths:
        path = pq if pq.is_absolute() else (repo_root / pq).resolve()
        if not path.is_file():
            raise FileNotFoundError(str(path))
        df = pd.read_parquet(path)
        for i in range(len(df)):
            raw = json.loads(str(df.iloc[i]["turns_json"]))
            turns = [DialogueTurn.model_validate(t) for t in raw]
            did = str(df.iloc[i]["dialogue_id"])
            rows.append((did, turns))
    return rows


def _label_one(
    client: OpenAI,
    model: str,
    did: str,
    turns: list[DialogueTurn],
    max_retries: int,
) -> dict | None:
    messages = build_inverter_messages(dialogue_turns=turns)
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": m["role"], "content": m["content"]} for m in messages],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            txt = (resp.choices[0].message.content or "").strip()
            raw_obj = json.loads(txt)
            raw_obj = repair_latent_z_json(raw_obj)
            z = validate_latent_z_json(raw_obj)
            return {
                "dialogue_id": did,
                "latent": z.model_dump(mode="json", exclude={"rationale"}),
            }
        except (json.JSONDecodeError, OSError, ValueError, TypeError, RuntimeError, openai.OpenAIError) as e:
            last_err = e
            wait = min(60.0, (2**attempt) + random.uniform(0, 0.5))
            time.sleep(wait)
    typer.echo(f"[warn] failed dialogue_id={did} after {max_retries} retries: {last_err}", err=True)
    return None


@app.command()
def main(
    repo_root: Path = typer.Option(Path("."), "--repo-root"),
    parquets: str = typer.Option(
        "data/processed/mathdial/train.parquet",
        "--parquets",
        help="Comma-separated parquet paths (relative to repo unless absolute).",
    ),
    out_jsonl: Path = typer.Option(Path("data/labels/latent_z_mathdial.jsonl"), "--out-jsonl"),
    limit: int = typer.Option(
        0,
        "--limit",
        help="Max new labels to write this run (0 = no cap). Resume skips already-done ids first.",
    ),
    model: str = typer.Option("gpt-4o-mini", "--model"),
    base_url: str = typer.Option("", "--base-url", help="Override API base URL (e.g. OpenRouter)."),
    resume: bool = typer.Option(True, "--resume/--no-resume", help="Skip dialogue_ids already present in out_jsonl."),
    workers: int = typer.Option(8, "--workers", help="Parallel API call threads."),
    max_retries: int = typer.Option(5, "--max-retries"),
) -> None:
    load_dotenv(repo_root / ".env")

    api_key = os.environ.get("OPENAI_API_KEY") or ""
    if not api_key:
        typer.echo("[err] OPENAI_API_KEY not set", err=True)
        raise typer.Exit(code=1)

    effective_base_url = base_url.strip() or os.environ.get("OPENROUTER_BASE_URL", "") or None

    extra_headers: dict[str, str] = {}
    if effective_base_url and "openrouter" in effective_base_url:
        if ref := os.environ.get("OPENROUTER_REFERRER"):
            extra_headers["HTTP-Referer"] = ref
        if title := os.environ.get("OPENROUTER_X_TITLE"):
            extra_headers["X-Title"] = title
        # OpenRouter uses its own key when hitting its API
        api_key = os.environ.get("OPENROUTER_API_KEY") or api_key

    client_kwargs: dict = {"api_key": api_key}
    if effective_base_url:
        client_kwargs["base_url"] = effective_base_url
    if extra_headers:
        client_kwargs["default_headers"] = extra_headers

    client = OpenAI(**client_kwargs)

    pq_list = [Path(p.strip()) for p in parquets.split(",") if p.strip()]
    out = repo_root / out_jsonl if not out_jsonl.is_absolute() else out_jsonl
    out.parent.mkdir(parents=True, exist_ok=True)

    done = _load_done_ids(out) if resume else set()
    all_rows = _iter_parquet_rows(repo_root, pq_list)
    pending = [(did, turns) for did, turns in all_rows if did not in done]
    if limit > 0:
        pending = pending[:limit]

    typer.echo(
        f"[info] parquets={len(pq_list)} total_rows={len(all_rows)} "
        f"already_done={len(done)} to_label={len(pending)} "
        f"workers={workers} model={model} base_url={effective_base_url or 'openai-default'}",
    )
    if not pending:
        typer.echo("[done] nothing to label")
        return

    n_ok = 0
    n_fail = 0

    with out.open("a", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=workers) as pool:
        futures: dict[Future, str] = {
            pool.submit(_label_one, client, model, did, turns, max_retries): did
            for did, turns in pending
        }
        with tqdm(total=len(futures), desc="label_latent_z", unit="dlg") as pbar:
            for fut in as_completed(futures):
                did = futures[fut]
                try:
                    rec = fut.result()
                except Exception as e:
                    typer.echo(f"[warn] unexpected exception dialogue_id={did}: {e}", err=True)
                    rec = None
                if rec is not None:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()
                    n_ok += 1
                else:
                    n_fail += 1
                pbar.update(1)
                pbar.set_postfix(ok=n_ok, fail=n_fail)

    typer.echo(f"[done] wrote {n_ok} new rows (fail={n_fail}) -> {out}")


if __name__ == "__main__":
    try:
        app()
    except KeyboardInterrupt:
        sys.exit(130)
