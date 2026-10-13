"""Math answers: eval-free parsing, bounded resources, equivalence, and extraction."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("sympy")

from epor.evaluation.mathcheck import (
    UnsafeOrInvalidAnswerError,
    answers_equivalent,
    extract_final_answer,
    parse_answer,
)


@pytest.mark.parametrize(
    ("candidate", "reference"),
    [
        ("1/2", "0.5"),
        ("\\frac{1}{2}", "2/4"),
        ("\\dfrac{3}{4}", "0.75"),
        ("2\\sqrt{2}", "sqrt(8)"),
        ("5\\sqrt{2}", "sqrt(50)"),
        ("1,234", "1234"),
        ("$12.50", "12.5"),
        ("50%", "1/2"),
        ("x^2+2x+1", "(x+1)**2"),
        ("2x+1", "(x+1)^2 - x^2"),
        ("2 \\cdot \\pi", "2pi"),
        ("-3", "-(3)"),
        ("60.", "60"),
    ],
)
def test_equivalent_answers(candidate: str, reference: str) -> None:
    assert answers_equivalent(candidate, reference)


@pytest.mark.parametrize(
    ("candidate", "reference"),
    [("3", "4"), ("x+1", "x+2"), ("1/3", "0.33"), ("", "1"), ("sqrt(2)", "1.414")],
)
def test_different_answers(candidate: str, reference: str) -> None:
    assert not answers_equivalent(candidate, reference)


MALICIOUS = [
    "__import__('os').system('touch {marker}')",
    "().__class__.__bases__[0].__subclasses__()",
    "exec('import os')",
    "eval('1')",
    "lambda: 1",
    "open('{marker}', 'w')",
    "x.__class__",
    "Symbol('x')",
    "1; import os",
    "[1, 2]",
]


@pytest.mark.parametrize("payload", MALICIOUS)
def test_malicious_inputs_are_rejected_without_execution(payload: str, tmp_path: Path) -> None:
    marker = tmp_path / "pwned"
    text = payload.format(marker=marker)
    with pytest.raises(UnsafeOrInvalidAnswerError):
        parse_answer(text)
    assert not answers_equivalent(text, "0")
    assert not marker.exists()


@pytest.mark.parametrize(
    "payload",
    [
        "9^9^9",
        "10^100000",
        "2**2**2**2**2**2",
        "(10^1000)^10",
        "1/0",
        "0^-1",
        "1" * 60,
        "(" * 40 + "1" + ")" * 40,
        "1+" * 200 + "1",
        "sqrt(-4)",
    ],
)
def test_resource_bounds_reject_expensive_or_undefined_answers(payload: str) -> None:
    with pytest.raises(UnsafeOrInvalidAnswerError):
        parse_answer(payload)


def test_parser_builds_exact_values() -> None:
    import sympy

    assert parse_answer("0.1 + 0.2") == sympy.Rational(3, 10)
    assert parse_answer("2^10") == 1024


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("work... #### 1,234", "1,234"),
        ("so \\boxed{\\frac{3}{4}} is it", "\\frac{3}{4}"),
        ("nested \\boxed{\\sqrt{\\frac{1}{4}}}", "\\sqrt{\\frac{1}{4}}"),
        ("The answer is 7. Then more.", "7"),
        ("Answer: x+1", "x+1"),
        ("first 3 then 5.5 done", "5.5"),
        ("no digits here", None),
    ],
)
def test_final_answer_extraction(text: str, expected: str | None) -> None:
    assert extract_final_answer(text) == expected
