"""LLM-assisted LatentZ labeling v3 — high-variance, evidence-discriminative silver labels.

Improvements over v2:
- Mandatory KC spread: >=3 KCs with mastery >0.7, >=2 with mastery <0.3
- Misconception: assign prob >=0.3 when student utterances show clear error patterns
- Metacog: must vary from defaults when behavioral cues exist
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

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

from iss.schema.grammar import validate_latent_z_json
from iss.schema.latent import DialogueTurn
from iss.schema.repair import repair_latent_z_json

app = typer.Typer(no_args_is_help=True)

INVERTER_SYSTEM_V3 = """\
You assess a student's latent cognitive state from a math tutoring dialogue.

Return EXACTLY this JSON structure (no markdown, no extra keys):
{
  "mastery": {"values": {"KC01": <float>, ..., "KC30": <float>}},
  "misconceptions": {"probs": {"M001": <float>, ..., "M070": <float>}},
  "metacog": {"monitoring_accuracy": <float>, "help_seeking_ratio": <float>,
              "confidence_correctness_gap": <float>, "hint_uptake": <float>}
}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
KC MASTERY (KC01..KC30, values in [0,1])
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
You MUST produce a spread that reflects the dialogue (not flat 0.5 everywhere):
  • At least 3 KCs with mastery >= 0.70 (clear strengths / correct reasoning)
  • At least 2 KCs with mastery <= 0.30 (clear weaknesses / errors)
  • Remaining KCs: 0.35–0.65 unless directly tested

Scale: 0.9=fluent correct | 0.7=mostly correct | 0.5=uncertain/untested |
       0.3=clear errors | 0.1=systematic failure

KC01=Whole-number place value | KC02=Integer ordering | KC03=Integer add/sub |
KC04=Integer multiply/divide | KC05=Factors/multiples | KC06=Prime factorization |
KC07=Fraction representation | KC08=Fraction comparison | KC09=Fraction add/sub |
KC10=Fraction multiply/divide | KC11=Decimal place value | KC12=Decimal operations |
KC13=Percent | KC14=Ratios | KC15=Proportional relationships | KC16=Linear expressions |
KC17=Equations | KC18=Inequalities | KC19=Exponents | KC20=Square roots |
KC21=Order of operations | KC22=Variables in word problems | KC23=Perimeter/area |
KC24=Angles | KC25=Coordinate plane | KC26=Slope | KC27=Patterns/sequences |
KC28=Data displays | KC29=Probability | KC30=Multi-step word problems

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MISCONCEPTIONS (M001..M070, values in [0,1])
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Default = 0.0. When student text shows an error pattern, assign >= 0.30 (not 0.02 hedges).
Expect 1–5 non-zero misconceptions per dialogue when errors appear.
  0.9 = explicit error in student words | 0.5 = answer consistent with pattern | 0.3 = plausible

M001=Add_denominators | M002=Ignore_common_denominator | M003=Invert_when_multiplying |
M004=Forget_invert_division | M005=Double_invert | M006=Larger_denominator_larger_value |
M007=Longer_is_bigger | M008=Cancel_across_addition | M009=Cancel_unlike_factors |
M010=Mixed_number_add_split_wrong | M011=Improper_to_mixed_wrong | M012=Half_of_fraction_doubles_denominator |
M013=Percent_as_decimal_shift_wrong | M014=Percent_add_to_base_wrong | M015=Ratio_add_denominators |
M016=Ratio_total_confusion | M017=Unit_rate_invert | M018=Scale_only_one_side |
M019=Fraction_equals_decimal_random | M020=Simplify_early_wrong | M021=Neg_plus_neg_positive |
M022=Subtract_negative_wrong | M023=Neg_times_neg_negative | M024=Neg_div_pos_negative_wrong |
M025=Absolute_value_changes_sign_always | M026=Number_line_direction | M027=PEMDAS_left_to_right_ignore |
M028=Implicit_mult_priority_wrong | M029=Exponent_distribute | M030=Exponent_multiply_add |
M031=Sqrt_add_linear | M032=Combine_unlike_terms | M033=Drop_exponent_on_substitute |
M034=Negative_outside_paren_dist_wrong | M035=Divide_cancel_x | M036=Square_both_sides_extraneous |
M037=Inequality_flip_forget | M038=Inequality_flip_always | M039=Linear_slope_sign |
M040=Intercept_swap | M041=Distribute_power | M042=Move_term_across_equals |
M043=Divide_partial_equation | M044=Multiply_one_side | M045=Clear_denominator_drop |
M046=Proportion_cross_mult_wrong | M047=System_substitute_partial | M048=Quadratic_no_constant |
M049=Factoring_drop_middle | M050=Percent_of_increase_wrong_base | M051=Percent_multiply_twice |
M052=Unit_cost_round_early | M053=Scale_model_area_linear | M054=Mixture_add_concentrations |
M055=Speed_avg_arithmetic_mean | M056=Map_scale_fraction_invert | M057=Area_perimeter_confusion |
M058=Triangle_area_no_half | M059=Angle_sum_parallel_misidentify | M060=Circle_pi_diameter_mix |
M061=Pythagorean_add_legs | M062=Volume_multiply_all_dims_twice | M063=Exponent_add_any_base |
M064=Negative_exponent_positive | M065=Scientific_notation_shift_count | M066=Root_square_cancel_always |
M067=Cube_root_even_properties | M068=Zero_exponent_one_exception | M069=Mean_include_repeated_counts_wrong |
M070=Probability_gt_one

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
METACOG (must reflect dialogue; avoid all-0.5 unless no cues)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
monitoring_accuracy [0,1]: catches own errors / self-corrects
help_seeking_ratio [0,1]: asks for help, hints, says "don't know"
confidence_correctness_gap [-1,1]: overconfident (+) vs underconfident (-)
hint_uptake [0,1]: uses tutor hints in later turns
"""


def _transcript_lines(turns: list[DialogueTurn]) -> str:
    lines: list[str] = []
    for t in turns:
        tag = "Tutor" if t.speaker == "tutor" else "Student"
        lines.append(f"{tag}: {t.text}")
    return "\n".join(lines)


def _build_messages_v3(question: str | None, turns: list[DialogueTurn]) -> list[dict[str, str]]:
    prefix = f"Math problem: {question}\n\n" if question else ""
    user = prefix + "Dialogue:\n" + _transcript_lines(turns) + "\n\nProvide LatentZ JSON only."
    return [
        {"role": "system", "content": INVERTER_SYSTEM_V3},
        {"role": "user", "content": user},
    ]


def _load_done_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    done: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            done.add(str(json.loads(line)["dialogue_id"]))
        except (json.JSONDecodeError, KeyError):
            continue
    return done


def _iter_parquet_rows(
    rr: Path, parquet_paths: list[Path]
) -> list[tuple[str, str | None, list[DialogueTurn]]]:
    rows: list[tuple[str, str | None, list[DialogueTurn]]] = []
    for pq in parquet_paths:
        path = pq if pq.is_absolute() else (rr / pq).resolve()
        if not path.is_file():
            raise FileNotFoundError(str(path))
        df = pd.read_parquet(path)
        for i in range(len(df)):
            row = df.iloc[i]
            raw_turns: list[dict] = json.loads(str(row["turns_json"]))
            turns = [DialogueTurn.model_validate(t) for t in raw_turns]
            did = str(row["dialogue_id"])
            question: str | None = None
            if "metadata_json" in row and row["metadata_json"]:
                try:
                    meta = json.loads(str(row["metadata_json"]))
                    question = meta.get("question") or None
                except (json.JSONDecodeError, TypeError):
                    pass
            rows.append((did, question, turns))
    return rows


def _label_one(
    client: OpenAI,
    model: str,
    did: str,
    question: str | None,
    turns: list[DialogueTurn],
    max_retries: int,
) -> dict | None:
    messages = _build_messages_v3(question, turns)
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
                "label_version": "v3",
            }
        except (json.JSONDecodeError, OSError, ValueError, TypeError, RuntimeError, openai.OpenAIError) as e:
            last_err = e
            time.sleep(min(60.0, (2**attempt) + random.uniform(0, 0.5)))
    typer.echo(f"[warn] failed dialogue_id={did}: {last_err}", err=True)
    return None


@app.command()
def main(
    repo_root_arg: Path = typer.Option(Path("."), "--repo-root"),
    parquets: str = typer.Option(
        "data/processed/mathdial/train.parquet,data/processed/mathdial/test.parquet",
        "--parquets",
    ),
    out_jsonl: Path = typer.Option(Path("data/labels/latent_z_silver_v3.jsonl"), "--out-jsonl"),
    limit: int = typer.Option(0, "--limit"),
    model: str = typer.Option("gpt-4o-mini", "--model"),
    base_url: str = typer.Option("", "--base-url"),
    resume: bool = typer.Option(True, "--resume/--no-resume"),
    workers: int = typer.Option(16, "--workers"),
    max_retries: int = typer.Option(5, "--max-retries"),
) -> None:
    rr = repo_root_arg.resolve()
    load_dotenv(rr / ".env")

    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY") or ""
    if not api_key:
        typer.echo("[err] OPENAI_API_KEY or OPENROUTER_API_KEY required", err=True)
        raise typer.Exit(code=1)

    effective_base_url = base_url.strip() or os.environ.get("OPENROUTER_BASE_URL", "") or None
    extra_headers: dict[str, str] = {}
    if effective_base_url and "openrouter" in effective_base_url:
        if ref := os.environ.get("OPENROUTER_REFERRER"):
            extra_headers["HTTP-Referer"] = ref
        if title := os.environ.get("OPENROUTER_X_TITLE"):
            extra_headers["X-Title"] = title
        api_key = os.environ.get("OPENROUTER_API_KEY") or api_key

    client_kwargs: dict = {"api_key": api_key}
    if effective_base_url:
        client_kwargs["base_url"] = effective_base_url
    if extra_headers:
        client_kwargs["default_headers"] = extra_headers
    client = OpenAI(**client_kwargs)

    pq_list = [Path(p.strip()) for p in parquets.split(",") if p.strip()]
    out = rr / out_jsonl if not out_jsonl.is_absolute() else out_jsonl
    out.parent.mkdir(parents=True, exist_ok=True)

    done = _load_done_ids(out) if resume else set()
    all_rows = _iter_parquet_rows(rr, pq_list)
    pending = [(did, q, turns) for did, q, turns in all_rows if did not in done]
    if limit > 0:
        pending = pending[:limit]

    typer.echo(
        f"[info] v3 labeling total={len(all_rows)} done={len(done)} pending={len(pending)} "
        f"workers={workers} model={model}"
    )
    if not pending:
        typer.echo("[done] nothing to label")
        return

    n_ok = n_fail = 0
    with out.open("a", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=workers) as pool:
        futures: dict[Future, str] = {
            pool.submit(_label_one, client, model, did, q, turns, max_retries): did
            for did, q, turns in pending
        }
        with tqdm(total=len(futures), desc="label_v3", unit="dlg") as pbar:
            for fut in as_completed(futures):
                try:
                    rec = fut.result()
                except Exception as e:
                    typer.echo(f"[warn] {futures[fut]}: {e}", err=True)
                    rec = None
                if rec is not None:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()
                    n_ok += 1
                else:
                    n_fail += 1
                pbar.update(1)
                pbar.set_postfix(ok=n_ok, fail=n_fail)

    typer.echo(f"[done] wrote {n_ok} rows (fail={n_fail}) -> {out}")


if __name__ == "__main__":
    try:
        app()
    except KeyboardInterrupt:
        sys.exit(130)
