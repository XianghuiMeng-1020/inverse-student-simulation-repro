"""Catalogue of |M|=70 common math misconceptions for multi-label diagnosis.

IDs are stable (`M001`..`M070`) for training, retrieval indices, and paper tables.
Items synthesize widely documented learner errors (fractions, integers,
algebra, geometry, ratios) aligned with middle-grades tutoring corpora.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import BaseModel, Field


class MisconceptionItem(BaseModel):
    id: str = Field(pattern=r"^M\d{3}$")
    name: str
    description: str
    example_wrong: str


def _spec() -> tuple[tuple[str, str, str], ...]:
    """(name, description, example_wrong) repeated 70 times."""
    return (
        # Fractions / rational numbers (1-20)
        (
            "Add_denominators",
            "Adds fractions by adding numerators and denominators separately.",
            "1/2 + 1/3 = 2/5",
        ),
        (
            "Ignore_common_denominator",
            "Adds numerators without adjusting denominators.",
            "1/2 + 1/4 = 2/6",
        ),
        (
            "Invert_when_multiplying",
            "Inverts a fraction incorrectly during multiplication.",
            "(2/3)*(4/5) = (3/2)*(4/5)",
        ),
        (
            "Forget_invert_division",
            "Divides fractions without inverting the divisor.",
            "(2/3) / (4/5) = (2/3)*(4/5)",
        ),
        (
            "Double_invert",
            "Inverts both fractions when dividing.",
            "Flip both fractions then multiply",
        ),
        (
            "Larger_denominator_larger_value",
            "Assumes a larger denominator makes the fraction larger.",
            "Claims 1/8 > 1/3",
        ),
        (
            "Longer_is_bigger",
            "Assumes more digits in numerator/denominator implies larger value.",
            "Claims 3/10 > 1/2 because 10 is bigger",
        ),
        (
            "Cancel_across_addition",
            "Cancels terms across a plus sign.",
            "(x+2)/(x+3) → 2/3 by 'canceling x'",
        ),
        (
            "Cancel_unlike_factors",
            "Cancels unrelated factors incorrectly.",
            "(2x+3)/(x) → 2+3",
        ),
        (
            "Mixed_number_add_split_wrong",
            "Adds whole and fractional parts without regrouping.",
            "2 3/4 + 1 3/4 = 3 6/8 without simplifying/regrouping",
        ),
        (
            "Improper_to_mixed_wrong",
            "Converts improper fractions to mixed numbers incorrectly.",
            "7/3 = 3 1/3",
        ),
        (
            "Half_of_fraction_doubles_denominator",
            "Takes half of a fraction by halving denominator only.",
            "Half of 2/3 is 2/6",
        ),
        (
            "Percent_as_decimal_shift_wrong",
            "Moves decimal point wrong direction when converting percent.",
            "3% = 0.3",
        ),
        (
            "Percent_add_to_base_wrong",
            "Adds percents without considering compounding/base.",
            "50% off then 20% off = 70% off total",
        ),
        (
            "Ratio_add_denominators",
            "Treats ratio like fractions addition incorrectly.",
            "2:3 + 1:3 = 3:6",
        ),
        (
            "Ratio_total_confusion",
            "Confuses part:part with part:whole.",
            "2:3 means 2 boys out of 3 students",
        ),
        (
            "Unit_rate_invert",
            "Inverts units in rate problems.",
            "40 miles / 2 hours → 2 miles per hour",
        ),
        (
            "Scale_only_one_side",
            "Scales only one term in equivalent ratios.",
            "2/3 = 4/5 by changing only numerator",
        ),
        (
            "Fraction_equals_decimal_random",
            "Equates unrelated fraction/decimal pairs.",
            "1/5 = 0.15",
        ),
        (
            "Simplify_early_wrong",
            "Simplifies intermediate results in a way that breaks equivalence.",
            "Changes only one term then continues",
        ),
        # Integers & signed arithmetic (21-26)
        (
            "Neg_plus_neg_positive",
            "Claims two negatives sum to a positive without rule.",
            "-3 + (-2) = +5",
        ),
        (
            "Subtract_negative_wrong",
            "Subtracting a negative decreases incorrectly.",
            "5 - (-2) = 3",
        ),
        (
            "Neg_times_neg_negative",
            "Wrong sign rule for multiplication.",
            "(-2)(-3) = -6",
        ),
        (
            "Neg_div_pos_negative_wrong",
            "Wrong sign for division.",
            "(-8)/2 = +4",
        ),
        (
            "Absolute_value_changes_sign_always",
            "Misuses absolute value as 'make negative positive' everywhere.",
            "|-3 + -2| = |3| + |2|",
        ),
        (
            "Number_line_direction",
            "Thinks numbers to the left are always greater when negatives appear.",
            "-1 < -8",
        ),
        # Order of operations / structure (27-31)
        (
            "PEMDAS_left_to_right_ignore",
            "Does multiplication before division always regardless of order.",
            "8/2*2 = 2",
        ),
        (
            "Implicit_mult_priority_wrong",
            "Ignores implied grouping in expressions like 2(3+4).",
            "2(3+4) = 2*3 + 4",
        ),
        (
            "Exponent_distribute",
            "Distributes exponent over addition.",
            "(x+2)^2 = x^2 + 4",
        ),
        (
            "Exponent_multiply_add",
            "Adds exponents when bases differ.",
            "2^3 * 3^2 = 5^5",
        ),
        (
            "Sqrt_add_linear",
            "Square root distributes over addition.",
            "sqrt(a+b) = sqrt(a)+sqrt(b)",
        ),
        # Algebra & expressions (32-41)
        (
            "Combine_unlike_terms",
            "Combines unlike variable terms.",
            "3x + 2y = 5xy",
        ),
        (
            "Drop_exponent_on_substitute",
            "Forgets exponent when substituting.",
            "If x=2, x^2 becomes 4 but student writes 2",
        ),
        (
            "Negative_outside_paren",
            "Distributes negative incorrectly.",
            "-(2x-3) = -2x-3",
        ),
        (
            "Divide_cancel_x",
            "Divides by x and drops solution x=0 without checking.",
            "x^2 = x ⇒ x = 1 only",
        ),
        (
            "Square_both_sides_extraneous",
            "Squares both sides and accepts all roots without checking.",
            "sqrt(x)=-2 gives x=4 as valid",
        ),
        (
            "Inequality_flip_forget",
            "Multiplies inequality by negative and forgets flip.",
            "-2x < 4 ⇒ x < -2",
        ),
        (
            "Inequality_flip_always",
            "Flips inequality even when multiplying by positive.",
            "2x > 6 ⇒ x < 3",
        ),
        (
            "Linear_slope_sign",
            "Misinterprets slope sign direction.",
            "Positive slope means line goes down left-to-right",
        ),
        (
            "Intercept_swap",
            "Swaps slope and intercept roles.",
            "y=2x+3 has slope 3",
        ),
        (
            "Distribute_power",
            "Distributes exponent over multiplication incorrectly.",
            "(ab)^2 = a^2 + b^2",
        ),
        # Equations & solving (42-49)
        (
            "Move_term_across_equals",
            "Moves a term across equals without changing sign.",
            "x+5=12 ⇒ x=12+5",
        ),
        (
            "Divide_partial_equation",
            "Divides only one side by a quantity.",
            "2x=8 ⇒ x=8",
        ),
        (
            "Multiply_one_side",
            "Multiplies only one side to clear fractions.",
            "x/2=3 ⇒ x=3",
        ),
        (
            "Clear_denominator_drop",
            "Clears denominators but drops factors.",
            "Multiply both sides by x but lose x=0",
        ),
        (
            "Proportion_cross_mult_wrong",
            "Cross-multiplies in non-proportional settings.",
            "Uses cross multiply on sum equation",
        ),
        (
            "System_substitute_partial",
            "Substitutes but leaves mixed variables incorrectly.",
            "Substitutes y yet leaves y in one term only",
        ),
        (
            "Quadratic_no_constant",
            "Forgets constant term when expanding binomial squared.",
            "(x+3)^2 = x^2 + 9",
        ),
        (
            "Factoring_drop_middle",
            "Factors x^2+5x+6 as (x+6)(x+1) without checking middle term.",
            "Wrong middle coefficient",
        ),
        # Ratios / proportions / percent (50-56)
        (
            "Percent_of_increase_wrong_base",
            "Uses wrong base for percent change.",
            "Increase from 40 to 50: 50/40 as percent change",
        ),
        (
            "Percent_multiply_twice",
            "Applies percent twice incorrectly.",
            "30% then 20% off as 50% off",
        ),
        (
            "Unit_cost_round_early",
            "Rounds too early in repeated unit pricing.",
            "Rounds each item then sums causing large error",
        ),
        (
            "Scale_model_area_linear",
            "Uses linear scale factor for area.",
            "Scale 1:10 so area scales 1:10",
        ),
        (
            "Mixture_add_concentrations",
            "Adds percents directly in mixture problems.",
            "30% acid + 20% acid = 50% acid mixture",
        ),
        (
            "Speed_avg_arithmetic_mean",
            "Averages speeds without harmonic mean context.",
            "60 mph there and 40 mph back means 50 mph average for same distance",
        ),
        (
            "Map_scale_fraction_invert",
            "Inverts map scale ratio.",
            "1 inch : 10 miles means 10 inches is 1 mile",
        ),
        # Geometry / measurement (57-62)
        (
            "Area_perimeter_confusion",
            "Confuses area and perimeter formulas.",
            "Uses 4s for area of square side s",
        ),
        (
            "Triangle_area_no_half",
            "Uses base*height without 1/2.",
            "A = bh",
        ),
        (
            "Angle_sum_parallel_misidentify",
            "Misidentifies corresponding/alternate interior angles.",
            "Claims alternate interior are supplementary always",
        ),
        (
            "Circle_pi_diameter_mix",
            "Confuses radius and diameter in circumference.",
            "C = pi*r",
        ),
        (
            "Pythagorean_add_legs",
            "Adds legs to get hypotenuse.",
            "3-4-5 triangle: hypotenuse is 3+4",
        ),
        (
            "Volume_multiply_all_dims_twice",
            "Double-counts a dimension in volume.",
            "V = l*w*w",
        ),
        # Exponents / roots / scientific notation (63-68)
        (
            "Exponent_add_any_base",
            "Adds exponents with different bases.",
            "2^3 * 3^2 = 5^5",
        ),
        (
            "Negative_exponent_positive",
            "Treats negative exponent as making base negative only.",
            "2^-3 = -8",
        ),
        (
            "Scientific_notation_shift_count",
            "Counts decimal shifts incorrectly.",
            "3.2*10^4 = 0.00032",
        ),
        (
            "Root_square_cancel_always",
            "Assumes sqrt(x^2)=x for all x.",
            "sqrt((-3)^2) = -3 only",
        ),
        (
            "Cube_root_even_properties",
            "Applies even-root rules to cube roots incorrectly.",
            "cube root of -8 is undefined",
        ),
        (
            "Zero_exponent_one_exception",
            "Claims 0^0 = 1 always in middle-school simplifications without domain.",
            "Simplifies 0^0 in invalid contexts",
        ),
        # Data / probability intro (69-70)
        (
            "Mean_include_repeated_counts_wrong",
            "Mis-handles frequency weighting in mean.",
            "Averages distinct outcomes ignoring counts",
        ),
        (
            "Probability_gt_one",
            "Adds probabilities for non-mutually exclusive events without correction.",
            "P(A or B) = P(A)+P(B) always",
        ),
    )


@lru_cache(maxsize=1)
def load_misconceptions() -> tuple[MisconceptionItem, ...]:
    rows: list[MisconceptionItem] = []
    for i, (name, desc, ex) in enumerate(_spec(), start=1):
        mid = f"M{i:03d}"
        rows.append(
            MisconceptionItem(
                id=mid,
                name=name,
                description=desc,
                example_wrong=ex,
            )
        )
    if len(rows) != 70:
        msg = f"Expected 70 misconceptions, got {len(rows)}"
        raise RuntimeError(msg)
    return tuple(rows)


def get_misconception_ids() -> tuple[str, ...]:
    return tuple(m.id for m in load_misconceptions())


def misconception_by_id() -> dict[str, MisconceptionItem]:
    return {m.id: m for m in load_misconceptions()}
