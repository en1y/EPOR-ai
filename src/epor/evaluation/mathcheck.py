"""Final-answer extraction and SymPy equivalence checking without ``eval``.

Safety reasoning
----------------
Model outputs and benchmark answer keys are untrusted text.  SymPy's own
string entry points (``sympify``, ``parse_expr``, ``S("...")``) compile the
string to Python and call ``eval`` on it, even with ``evaluate=False``; a
string such as ``__import__('os').system('...')`` would run.  This module
never hands a string to any of them.  Instead a small recursive-descent parser
recognizes a closed grammar and builds SymPy objects directly from tokens:

* numbers (integers, decimals as exact rationals), single-letter symbols,
  ``pi``/``\\pi``, ``+ - * / ^ **``, parentheses and braces, implicit
  multiplication, ``sqrt(...)``/``\\sqrt{...}``, ``\\frac{a}{b}``, and a
  trailing ``%``;
* anything else (identifiers longer than one letter, underscores, quotes,
  dots outside numbers, brackets, commas outside digit groups) is rejected.

Because SymPy evaluates arithmetic eagerly, the parser also bounds the cost of
what it builds: input length, token count, nesting depth, integer-literal
length, and every power (the exponent must be a small number or symbolic, and
a numeric power may not produce more than ``MAX_POWER_DIGITS`` digits), so an
answer like ``9^9^9`` is rejected instead of exhausting memory.  Non-finite
results (``1/0``) are rejected too.  Equivalence is exact: ``simplify(a - b)
== 0`` on two bounded expressions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

MAX_ANSWER_CHARS = 256
MAX_TOKENS = 128
MAX_DEPTH = 32
MAX_INTEGER_DIGITS = 50
MAX_EXPONENT = 1_000
MAX_POWER_DIGITS = 5_000


class UnsafeOrInvalidAnswerError(ValueError):
    """The answer text is outside the safe grammar or its resource bounds."""


_TOKEN = re.compile(
    r"\s*(?:"
    r"(?P<number>\d+(?:\.\d+)?|\.\d+)"
    r"|(?P<command>\\[a-zA-Z]+)"
    r"|(?P<name>[A-Za-z]+)"
    r"|(?P<op>\*\*|[-+*/^(){}%])"
    r")"
)
_REPLACEMENTS = (
    (re.compile(r"\\(?:left|right|displaystyle|!|,|;|:| )"), ""),
    (re.compile(r"\\[dt]frac"), r"\\frac"),
    (re.compile(r"\\(?:cdot|times)"), "*"),
    (re.compile(r"\\%"), "%"),
    (re.compile(r"\$"), ""),
    (re.compile(r"(?<=\d),(?=\d{3}(?:\D|$))"), ""),
)
_FUNCTIONS = frozenset({"sqrt"})
_CONSTANTS = frozenset({"pi"})


@dataclass(slots=True)
class _Token:
    kind: str
    text: str


def _tokenize(text: str) -> list[_Token]:
    tokens: list[_Token] = []
    position = 0
    stripped = text.strip()
    while position < len(stripped):
        match = _TOKEN.match(stripped, position)
        if match is None or match.end() == position:
            raise UnsafeOrInvalidAnswerError(f"unsupported character at {position}")
        position = match.end()
        kind = match.lastgroup
        assert kind is not None
        value = match.group(kind)
        if kind == "command":
            name = value[1:]
            if name == "frac":
                tokens.append(_Token("frac", name))
            elif name in _FUNCTIONS:
                tokens.append(_Token("function", name))
            elif name in _CONSTANTS:
                tokens.append(_Token("constant", name))
            else:
                raise UnsafeOrInvalidAnswerError(f"unsupported command \\{name}")
        elif kind == "name":
            if value in _FUNCTIONS:
                tokens.append(_Token("function", value))
            elif value in _CONSTANTS:
                tokens.append(_Token("constant", value))
            elif len(value) == 1:
                tokens.append(_Token("symbol", value))
            else:
                raise UnsafeOrInvalidAnswerError("multi-letter identifiers are not supported")
        else:
            tokens.append(_Token(kind, value))
        if len(tokens) > MAX_TOKENS:
            raise UnsafeOrInvalidAnswerError("answer has too many tokens")
    if not tokens:
        raise UnsafeOrInvalidAnswerError("empty answer")
    return tokens


class _Parser:
    def __init__(self, tokens: list[_Token]) -> None:
        import sympy

        self.sympy = sympy
        self.tokens = tokens
        self.position = 0
        self.depth = 0

    def peek(self) -> _Token | None:
        return self.tokens[self.position] if self.position < len(self.tokens) else None

    def take(self, kind: str, text: str | None = None) -> _Token:
        token = self.peek()
        if token is None or token.kind != kind or (text is not None and token.text != text):
            raise UnsafeOrInvalidAnswerError(f"expected {text or kind}")
        self.position += 1
        return token

    def at_op(self, *texts: str) -> bool:
        token = self.peek()
        return token is not None and token.kind == "op" and token.text in texts

    def enter(self) -> None:
        self.depth += 1
        if self.depth > MAX_DEPTH:
            raise UnsafeOrInvalidAnswerError("answer is nested too deeply")

    def leave(self) -> None:
        self.depth -= 1

    def parse(self) -> Any:
        value = self.expression()
        if self.peek() is not None:
            raise UnsafeOrInvalidAnswerError("unexpected trailing input")
        return value

    def expression(self) -> Any:
        self.enter()
        value = self.term()
        while self.at_op("+", "-"):
            operator = self.take("op").text
            right = self.term()
            value = value + right if operator == "+" else value - right
        self.leave()
        return value

    def starts_atom(self) -> bool:
        token = self.peek()
        if token is None:
            return False
        return token.kind in {"number", "symbol", "constant", "function", "frac"} or (
            token.kind == "op" and token.text in {"(", "{"}
        )

    def term(self) -> Any:
        value = self.unary()
        while True:
            if self.at_op("*", "/"):
                operator = self.take("op").text
                right = self.unary()
                if operator == "*":
                    value = value * right
                else:
                    if right == 0:
                        raise UnsafeOrInvalidAnswerError("division by zero")
                    value = value / right
            elif self.starts_atom():
                value = value * self.unary()
            else:
                return value

    def unary(self) -> Any:
        if self.at_op("+", "-"):
            operator = self.take("op").text
            self.enter()
            operand = self.unary()
            self.leave()
            return operand if operator == "+" else -operand
        return self.power()

    def power(self) -> Any:
        base = self.postfix(self.atom())
        if self.at_op("^", "**"):
            self.take("op")
            self.enter()
            exponent = self.unary()
            self.leave()
            return self.bounded_power(base, exponent)
        return base

    def postfix(self, value: Any) -> Any:
        if self.at_op("%"):
            self.take("op")
            return value / 100
        return value

    def bounded_power(self, base: Any, exponent: Any) -> Any:
        sympy = self.sympy
        if exponent.is_number:
            if not exponent.is_finite or abs(exponent) > MAX_EXPONENT:
                raise UnsafeOrInvalidAnswerError("exponent is too large")
            if base.is_number:
                magnitude = abs(base)
                if magnitude == 0 and exponent.is_negative:
                    raise UnsafeOrInvalidAnswerError("division by zero")
                if magnitude != 0:
                    log10 = float(sympy.log(magnitude, 10).evalf(15))
                    if abs(float(exponent)) * abs(log10) > MAX_POWER_DIGITS:
                        raise UnsafeOrInvalidAnswerError("power result is too large")
        return sympy.Pow(base, exponent)

    def group(self, opening: str, closing: str) -> Any:
        self.take("op", opening)
        value = self.expression()
        self.take("op", closing)
        return value

    def atom(self) -> Any:
        sympy = self.sympy
        token = self.peek()
        if token is None:
            raise UnsafeOrInvalidAnswerError("unexpected end of answer")
        if token.kind == "number":
            self.position += 1
            integer_part = token.text.split(".")[0]
            if len(integer_part) > MAX_INTEGER_DIGITS or len(token.text) > 2 * MAX_INTEGER_DIGITS:
                raise UnsafeOrInvalidAnswerError("number literal is too long")
            return sympy.Rational(token.text)
        if token.kind == "symbol":
            self.position += 1
            return sympy.Symbol(token.text)
        if token.kind == "constant":
            self.position += 1
            return sympy.pi
        if token.kind == "function":
            self.position += 1
            opening = "{" if self.at_op("{") else "("
            argument = self.group(opening, "}" if opening == "{" else ")")
            if argument.is_number and argument.is_negative:
                raise UnsafeOrInvalidAnswerError("square root of a negative number")
            return sympy.sqrt(argument)
        if token.kind == "frac":
            self.position += 1
            numerator = self.group("{", "}")
            denominator = self.group("{", "}")
            if denominator == 0:
                raise UnsafeOrInvalidAnswerError("division by zero")
            return numerator / denominator
        if token.kind == "op" and token.text in {"(", "{"}:
            self.enter()
            value = self.group(token.text, ")" if token.text == "(" else "}")
            self.leave()
            return value
        raise UnsafeOrInvalidAnswerError(f"unexpected token {token.text!r}")


def normalize_answer_text(text: str) -> str:
    """Strip LaTeX decoration, dollar signs, and thousands separators."""

    value = text.strip()
    for pattern, replacement in _REPLACEMENTS:
        value = pattern.sub(replacement, value)
    return value.strip().rstrip(".").strip()


def parse_answer(text: str) -> Any:
    """Parse untrusted answer text into a bounded SymPy expression (never ``eval``)."""

    if len(text) > MAX_ANSWER_CHARS:
        raise UnsafeOrInvalidAnswerError("answer text is too long")
    value = _Parser(_tokenize(normalize_answer_text(text))).parse()
    if value.has(*_non_finite()):
        raise UnsafeOrInvalidAnswerError("answer is not finite")
    return value


def _non_finite() -> tuple[Any, ...]:
    import sympy

    return (sympy.zoo, sympy.oo, -sympy.oo, sympy.nan)


def answers_equivalent(candidate: str, reference: str) -> bool:
    """``True`` when both parse safely and are symbolically equal; ``False`` otherwise."""

    import sympy

    try:
        left = parse_answer(candidate)
        right = parse_answer(reference)
    except UnsafeOrInvalidAnswerError:
        return False
    difference = left - right
    if difference == 0:
        return True
    return bool(sympy.simplify(difference) == 0)


_BOXED = "\\boxed{"
_GSM8K = re.compile(r"####\s*([^\n]+)")
_ANSWER_IS = re.compile(r"(?:answer is|answer:)\s*([^\n]+)", re.IGNORECASE)
_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?(?:/\d+)?")


def _last_boxed(text: str) -> str | None:
    start = text.rfind(_BOXED)
    if start < 0:
        return None
    depth = 1
    position = start + len(_BOXED)
    for index in range(position, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[position:index]
    return None


def extract_final_answer(text: str) -> str | None:
    """The final answer in a solution: ``####``, ``\\boxed{}``, "answer is", or last number."""

    matches = _GSM8K.findall(text)
    if matches:
        return matches[-1].strip()
    boxed = _last_boxed(text)
    if boxed is not None:
        return boxed.strip()
    stated = _ANSWER_IS.findall(text)
    if stated:
        return re.sub(r"\.\s.*$", "", stated[-1]).strip().rstrip(".")
    numbers = _NUMBER.findall(text)
    if numbers:
        return numbers[-1]
    return None


__all__ = [
    "MAX_ANSWER_CHARS",
    "UnsafeOrInvalidAnswerError",
    "answers_equivalent",
    "extract_final_answer",
    "normalize_answer_text",
    "parse_answer",
]
