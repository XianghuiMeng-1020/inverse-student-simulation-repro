"""Fixed K=30 knowledge-component ontology for ISS (middle-grades math).

The list is curriculum-aligned for MathDial-style multi-step reasoning.
Gold alignment on corpora is obtained via LLM-assisted tagging + manual audit
subset (paper protocol).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from pydantic import BaseModel, Field


class KnowledgeComponent(BaseModel):
    """One KC with short definitions and few-shot exemplars."""

    id: str = Field(pattern=r"^KC\d{2}$")
    name: str
    description: str
    examples: list[str] = Field(min_length=3, max_length=3)


_KC_ROWS: list[dict[str, Any]] = [
    {
        "id": "KC01",
        "name": "Whole number place value",
        "description": "Understanding tens/hundreds relationships and expanded form.",
        "examples": ["Write 5406 in expanded form.", "Which digit is in the hundreds place in 12,309?", "Round 347 to the nearest ten."],
    },
    {
        "id": "KC02",
        "name": "Integer ordering and comparison",
        "description": "Compare and order integers on the number line, including negatives.",
        "examples": ["Which is greater: -4 or -9?", "Order -2, 0, -7 from least to greatest.", "Plot -3.5 on a number line."],
    },
    {
        "id": "KC03",
        "name": "Integer addition and subtraction",
        "description": "Add/subtract integers with number-line and decomposition strategies.",
        "examples": ["Compute -8 + 5.", "Compute 3 - (-2).", "Explain why subtracting a negative increases value."],
    },
    {
        "id": "KC04",
        "name": "Integer multiplication and division",
        "description": "Sign rules and magnitude reasoning for integer multiply/divide.",
        "examples": ["Compute (-6)(-4).", "Compute 24 / (-3).", "Is the product of two negatives positive? Why?"],
    },
    {
        "id": "KC05",
        "name": "Factors, multiples, and divisibility",
        "description": "Find factors/multiples; use divisibility tests in reasoning.",
        "examples": ["List all factors of 36.", "Find LCM(6, 8).", "Is 129 divisible by 3? Explain."],
    },
    {
        "id": "KC06",
        "name": "Prime factorization and GCD/LCM",
        "description": "Prime factor trees; GCD/LCM from prime factorization.",
        "examples": ["Prime factorize 180.", "Find GCD(84, 126).", "Use prime factorization to find LCM(12, 18)."],
    },
    {
        "id": "KC07",
        "name": "Fraction representation",
        "description": "Part-whole meaning; equivalent fractions; simplifying.",
        "examples": ["Shade 3/8 of a rectangle.", "Find an equivalent fraction to 2/5 with denominator 15.", "Simplify 18/24."],
    },
    {
        "id": "KC08",
        "name": "Fraction comparison",
        "description": "Compare fractions via common denominators or benchmarks.",
        "examples": ["Which is larger: 3/7 or 2/5?", "Compare 5/12 to 1/2 without calculating exact decimals.", "Order 2/3, 3/5, 7/10."],
    },
    {
        "id": "KC09",
        "name": "Fraction addition/subtraction",
        "description": "Common denominator strategy; mixed numbers.",
        "examples": ["Compute 2/3 + 1/4.", "Compute 5 1/2 - 2 2/3.", "Explain why denominators must match to add."],
    },
    {
        "id": "KC10",
        "name": "Fraction multiplication/division",
        "description": "Multiply fractions; invert-and-multiply for division; scaling interpretation.",
        "examples": ["Compute (2/3)(9/10).", "Compute (3/4) / (2/5).", "What is half of two-thirds?"],
    },
    {
        "id": "KC11",
        "name": "Decimal place value and rounding",
        "description": "Tenths/hundredths; rounding decimals to a place.",
        "examples": ["Round 3.147 to the nearest tenth.", "Which is greater: 0.6 or 0.059?", "Write 4.07 in words."],
    },
    {
        "id": "KC12",
        "name": "Decimal operations",
        "description": "Add/subtract/multiply/divide decimals with place-value reasoning.",
        "examples": ["Compute 2.6 + 0.35.", "Compute 0.08 * 1.5.", "Compute 6.3 / 0.07."],
    },
    {
        "id": "KC13",
        "name": "Percent and percent change",
        "description": "Percent as per-100; percent increase/decrease.",
        "examples": ["What is 15% of 240?", "A price rises from $40 to $50. What is the percent increase?", "Convert 0.125 to a percent."],
    },
    {
        "id": "KC14",
        "name": "Ratios and equivalent ratios",
        "description": "Ratio language; tables; scaling to equivalent ratios.",
        "examples": ["Write the ratio of 8 boys to 12 girls in simplest form.", "If 2:3 scales to 10:?, find the missing value.", "Is 4/6 equivalent to 10/15?"],
    },
    {
        "id": "KC15",
        "name": "Proportional relationships",
        "description": "Unit rate; constant of proportionality y = kx.",
        "examples": ["If 5 miles takes 40 minutes, what is the unit rate in minutes per mile?", "Find k if y = kx passes through (2, 7).", "Is this table proportional? Explain."],
    },
    {
        "id": "KC16",
        "name": "Linear expressions",
        "description": "Terms, coefficients, like terms, distributing.",
        "examples": ["Simplify 3x + 5 - 2x + 1.", "Expand 4(2x - 3).", "Is 2(x+3) equivalent to 2x+3? Explain."],
    },
    {
        "id": "KC17",
        "name": "One-step and two-step equations",
        "description": "Solve ax + b = c using inverse operations; maintain equality.",
        "examples": ["Solve 3x = 12.", "Solve 2x - 5 = 11.", "Explain each step when solving 4x + 7 = 35."],
    },
    {
        "id": "KC18",
        "name": "Linear inequalities (one variable)",
        "description": "Solve and graph simple inequalities; flip inequality when multiplying by a negative.",
        "examples": ["Solve 2x + 1 < 9.", "Graph x ≥ -3 on a number line.", "Why do we flip the sign when multiplying both sides by -1?"],
    },
    {
        "id": "KC19",
        "name": "Exponents and scientific notation",
        "description": "Integer exponents; laws of exponents basics; scientific notation conversions.",
        "examples": ["Compute 2^5.", "Simplify x^3 * x^4.", "Write 0.0045 in scientific notation."],
    },
    {
        "id": "KC20",
        "name": "Square roots and perfect squares",
        "description": "Perfect squares; principal square root; estimation.",
        "examples": ["What is sqrt(81)?", "Between which two integers does sqrt(50) lie?", "Is sqrt(x^2) always x? Explain."],
    },
    {
        "id": "KC21",
        "name": "Order of operations (PEMDAS/GEMA)",
        "description": "Parentheses, exponents, multiplication/division, addition/subtraction conventions.",
        "examples": ["Compute 3 + 4 * 2.", "Compute (2 + 3)^2 - 4.", "Insert parentheses to make 2+3*4 equal 20."],
    },
    {
        "id": "KC22",
        "name": "Variables and expressions in word problems",
        "description": "Translate verbal descriptions into expressions/equations.",
        "examples": ["Write an expression for '5 more than twice a number'.", "Define a variable for a ticket problem.", "Translate: total cost is $3 per item plus $2 fee."],
    },
    {
        "id": "KC23",
        "name": "Perimeter and area (rectangles/triangles)",
        "description": "Compute perimeter/area; units; common formulas.",
        "examples": ["Find the area of a rectangle 7 cm by 4 cm.", "Find perimeter of a triangle with sides 3,4,5.", "Explain why area uses square units."],
    },
    {
        "id": "KC24",
        "name": "Angles and parallel lines (intro)",
        "description": "Complementary/supplementary; vertical angles; parallel line angle pairs.",
        "examples": ["Two angles are supplementary; one is 70°. Find the other.", "If two parallel lines are cut by a transversal, name equal angles.", "Find missing angle in a triangle sum problem."],
    },
    {
        "id": "KC25",
        "name": "Coordinate plane basics",
        "description": "Plot points; quadrants; distance along axis (intro).",
        "examples": ["Plot (-2, 5).", "Which quadrant contains (4, -1)?", "What is the distance from (0,0) to (0, 6)?"],
    },
    {
        "id": "KC26",
        "name": "Slope and rate of change (intro)",
        "description": "Slope from two points; interpret as rate.",
        "examples": ["Find slope through (1,2) and (3,8).", "Interpret slope in a distance-time context.", "Is slope constant in proportional relationships?"],
    },
    {
        "id": "KC27",
        "name": "Patterns and sequences (arithmetic)",
        "description": "Recognize arithmetic patterns; nth term intro.",
        "examples": ["Next term: 5, 8, 11, 14, …", "Find the 10th term of an arithmetic sequence starting at 4 with common difference 3.", "Write a rule for the pattern."],
    },
    {
        "id": "KC28",
        "name": "Data displays and central tendency",
        "description": "Mean/median/mode; dot plots/histograms (interpretive).",
        "examples": ["Find the mean of {2,7,7,10}.", "Which measure is best for skewed data? Why?", "Interpret a peak in a histogram."],
    },
    {
        "id": "KC29",
        "name": "Probability basics (intro)",
        "description": "Sample space; empirical vs theoretical probability; simple compound events.",
        "examples": ["Probability of rolling an even number on a fair die.", "If you draw a card from a shuffled deck, P(red)?", "Two coin flips: P(HT)?"],
    },
    {
        "id": "KC30",
        "name": "Multi-step word problem integration",
        "description": "Decompose problems requiring multiple operations/constraints.",
        "examples": ["Two-variable ticket problem with total people and total revenue.", "Distance-rate-time with unit conversion.", "Mixture problem with two concentrations."],
    },
]


@lru_cache(maxsize=1)
def load_knowledge_components() -> tuple[KnowledgeComponent, ...]:
    """Return the canonical KC tuple (length 30)."""
    return tuple(KnowledgeComponent.model_validate(row) for row in _KC_ROWS)


def get_kc_ids() -> tuple[str, ...]:
    """Return KC01..KC30 in order."""
    return tuple(kc.id for kc in load_knowledge_components())


def kc_by_id() -> dict[str, KnowledgeComponent]:
    """Map id -> KC."""
    return {kc.id: kc for kc in load_knowledge_components()}
