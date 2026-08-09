"""LLM-assisted LatentZ labeling v2 — discriminative prompt with KC/Misc descriptions.

Key improvements over v1:
- Full KC names in prompt (so LLM knows what KC01..KC30 mean)
- Misconception descriptions (so LLM can match to student errors)
- Math problem context included in user message
- Explicit default rules: KC mastery defaults to 0.5, Misc defaults to 0.0
- Forced sparse Misc output (expect 0-3 non-zero values)
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

# ---------------------------------------------------------------------------
# V2 system prompt — discriminative, evidence-based
# ---------------------------------------------------------------------------

INVERTER_SYSTEM_V2 = """\
You assess a student's latent cognitive state from a math tutoring dialogue.

Return EXACTLY this JSON structure (no markdown, no extra keys):
{
  "mastery": {"values": {"KC01": <float>, ..., "KC30": <float>}},
  "misconceptions": {"probs": {"M001": <float>, ..., "M070": <float>}},
  "metacog": {"monitoring_accuracy": <float>, "help_seeking_ratio": <float>,
              "confidence_correctness_gap": <float>, "hint_uptake": <float>}
}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
KC MASTERY RULES  (keys KC01..KC30, values 0.0–1.0)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Default = 0.5 (uncertain / not tested). Only deviate for KCs directly exercised in the problem:
  0.9 = Student clearly demonstrates correct reasoning for this KC
  0.7 = Mostly correct; only minor errors
  0.3 = Student makes clear errors on this KC
  0.1 = Student clearly fails; systematic errors or confusion

Knowledge Components:
KC01=Whole-number place value | KC02=Integer ordering/comparison | KC03=Integer add/subtract | KC04=Integer multiply/divide
KC05=Factors/multiples/divisibility | KC06=Prime factorization/GCD/LCM | KC07=Fraction representation | KC08=Fraction comparison
KC09=Fraction add/subtract | KC10=Fraction multiply/divide | KC11=Decimal place value/rounding | KC12=Decimal operations
KC13=Percent/percent change | KC14=Ratios/equivalent ratios | KC15=Proportional relationships | KC16=Linear expressions
KC17=One/two-step equations | KC18=Linear inequalities (one variable) | KC19=Exponents/scientific notation | KC20=Square roots/perfect squares
KC21=Order of operations (PEMDAS) | KC22=Variables/expressions in word problems | KC23=Perimeter/area (rectangles/triangles)
KC24=Angles/parallel lines | KC25=Coordinate plane basics | KC26=Slope/rate of change | KC27=Patterns/sequences (arithmetic)
KC28=Data displays/central tendency | KC29=Probability basics | KC30=Multi-step word problem integration

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MISCONCEPTION RULES  (keys M001..M070, values 0.0–1.0)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Default = 0.0 (not evidenced). Only assign > 0 with EXPLICIT evidence in student utterances:
  0.9 = Student utterances CLEARLY show this error pattern
  0.5 = Strong indirect evidence (wrong answer consistent with this pattern)

⚠ Expect at most 0–3 non-zero values. Resist the urge to assign 0.5 as a hedge.

M001=Add_denominators(adds num+denom separately, e.g. 1/2+1/3=2/5) | M002=Ignore_common_denominator(adds numerators only)
M003=Invert_when_multiplying(inverts fraction incorrectly in multiply) | M004=Forget_invert_division(divides without inverting divisor)
M005=Double_invert(inverts both fractions when dividing) | M006=Larger_denominator_larger_value(bigger denom=bigger fraction)
M007=Longer_is_bigger(more digits=larger) | M008=Cancel_across_addition(cancels across + sign)
M009=Cancel_unlike_factors | M010=Mixed_number_add_split_wrong(no regrouping)
M011=Improper_to_mixed_wrong | M012=Half_of_fraction_doubles_denominator
M013=Percent_as_decimal_shift_wrong | M014=Percent_add_to_base_wrong
M015=Ratio_add_denominators | M016=Ratio_total_confusion(part:part vs part:whole)
M017=Unit_rate_invert | M018=Scale_only_one_side(scales one term only in ratio)
M019=Fraction_equals_decimal_random | M020=Simplify_early_wrong
M021=Neg_plus_neg_positive(thinks -a + -b = positive) | M022=Subtract_negative_wrong
M023=Neg_times_neg_negative(wrong sign for neg×neg) | M024=Neg_div_pos_negative_wrong(wrong sign)
M025=Absolute_value_changes_sign_always | M026=Number_line_direction(left=greater confusion)
M027=PEMDAS_left_to_right_ignore(mult always before div) | M028=Implicit_mult_priority_wrong
M029=Exponent_distribute(a+b)^n=a^n+b^n | M030=Exponent_multiply_add(adds exponents with diff bases)
M031=Sqrt_add_linear(sqrt(a+b)=sqrt(a)+sqrt(b)) | M032=Combine_unlike_terms(x+x^2=2x^2)
M033=Drop_exponent_on_substitute | M034=Negative_outside_paren_dist_wrong
M035=Divide_cancel_x(drops x=0 solution) | M036=Square_both_sides_extraneous
M037=Inequality_flip_forget(doesn't flip when mult by neg) | M038=Inequality_flip_always(flips when mult by pos too)
M039=Linear_slope_sign | M040=Intercept_swap(swaps slope and intercept)
M041=Distribute_power | M042=Move_term_across_equals(no sign change)
M043=Divide_partial_equation(divides only one side) | M044=Multiply_one_side(multiplies one side to clear fractions)
M045=Clear_denominator_drop | M046=Proportion_cross_mult_wrong
M047=System_substitute_partial | M048=Quadratic_no_constant(forgets constant in (x+a)^2)
M049=Factoring_drop_middle | M050=Percent_of_increase_wrong_base
M051=Percent_multiply_twice | M052=Unit_cost_round_early
M053=Scale_model_area_linear(uses linear scale for area) | M054=Mixture_add_concentrations(adds percents directly)
M055=Speed_avg_arithmetic_mean | M056=Map_scale_fraction_invert
M057=Area_perimeter_confusion | M058=Triangle_area_no_half(forgets 1/2)
M059=Angle_sum_parallel_misidentify | M060=Circle_pi_diameter_mix(confuses radius/diameter)
M061=Pythagorean_add_legs(a+b=c) | M062=Volume_multiply_all_dims_twice
M063=Exponent_add_any_base | M064=Negative_exponent_positive(makes base negative only)
M065=Scientific_notation_shift_count | M066=Root_square_cancel_always(sqrt(x^2)=x always)
M067=Cube_root_even_properties | M068=Zero_exponent_one_exception
M069=Mean_include_repeated_counts_wrong | M070=Probability_gt_one(adds non-mutually-exclusive probs)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
METACOG ASSESSMENT  (all values in stated ranges)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
monitoring_accuracy [0,1]: Does student catch own errors / re-read carefully? 1=yes frequently; 0=never
help_seeking_ratio [0,1]: Does student frequently ask for help/hints? 1=very often; 0=always tries independently first
confidence_correctness_gap [-1,1]: Is student overconfident(+) or underconfident(-) relative to performance? 0=calibrated
hint_uptake [0,1]: Does student use tutor hints to improve? 1=always improves after hint; 0=ignores hints

If a metacog dimension lacks behavioral cues: monitoring_accuracy=0.5, help_seeking_ratio=0.5, confidence_correctness_gap=0.0, hint_uptake=0.5
"""


def _transcript_lines(turns: list[DialogueTurn]) -> str:
    lines: list[str] = []
    for t in turns:
        tag = "Tutor" if t.speaker == "tutor" else "Student"
        lines.append(f"{tag}: {t.text}")
    return "\n".join(lines)


def _build_messages_v2(
    question: str | None,
    turns: list[DialogueTurn],
) -> list[dict[str, str]]:
    prefix = f"Math problem: {question}\n\n" if question else ""
    user = prefix + "Dialogue:\n" + _transcript_lines(turns) + "\n\nProvide LatentZ JSON only."
    return [
        {"role": "system", "content": INVERTER_SYSTEM_V2},
        {"role": "user", "content": user},
    ]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

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
    repo_root: Path, parquet_paths: list[Path]
) -> list[tuple[str, str | None, list[DialogueTurn]]]:
    rows: list[tuple[str, str | None, list[DialogueTurn]]] = []
    for pq in parquet_paths:
        path = pq if pq.is_absolute() else (repo_root / pq).resolve()
        if not path.is_file():
            raise FileNotFoundError(str(path))
        df = pd.read_parquet(path)
        for i in range(len(df)):
            row = df.iloc[i]
            raw_turns: list[dict] = json.loads(str(row["turns_json"]))
            turns = [DialogueTurn.model_validate(t) for t in raw_turns]
            did = str(row["dialogue_id"])
            # Extract math problem from metadata if available
            question: str | None = None
            if "metadata_json" in row and row["metadata_json"]:
                try:
                    meta = json.loads(str(row["metadata_json"]))
                    question = meta.get("question") or None
                except (json.JSONDecodeError, TypeError):
                    pass
            rows.append((did, question, turns))
    return rows


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def _label_one(
    client: OpenAI,
    model: str,
    did: str,
    question: str | None,
    turns: list[DialogueTurn],
    max_retries: int,
) -> dict | None:
    messages = _build_messages_v2(question, turns)
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@app.command()
def main(
    repo_root_arg: Path = typer.Option(Path("."), "--repo-root"),
    parquets: str = typer.Option(
        "data/processed/mathdial/train.parquet,data/processed/mathdial/test.parquet",
        "--parquets",
        help="Comma-separated parquet paths.",
    ),
    out_jsonl: Path = typer.Option(Path("data/labels/latent_z_silver_v2.jsonl"), "--out-jsonl"),
    limit: int = typer.Option(0, "--limit", help="0 = no cap"),
    model: str = typer.Option("gpt-4o-mini", "--model"),
    base_url: str = typer.Option("", "--base-url"),
    resume: bool = typer.Option(True, "--resume/--no-resume"),
    workers: int = typer.Option(16, "--workers"),
    max_retries: int = typer.Option(5, "--max-retries"),
) -> None:
    rr = repo_root_arg.resolve()
    load_dotenv(rr / ".env")

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
            pool.submit(_label_one, client, model, did, q, turns, max_retries): did
            for did, q, turns in pending
        }
        with tqdm(total=len(futures), desc="label_v2", unit="dlg") as pbar:
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
