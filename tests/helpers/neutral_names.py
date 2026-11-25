"""Neutral-name scanner (E10-T01, FR-G01): lists finance terms in the modules that become `hsmmkit`.

A hit is `path:line:term`. Identifiers (names, attributes, arguments, function/class names, keyword
names, dict keys in literals) and text (string constants, docstrings, comments) are matched
case-insensitively as whole words after splitting snake and camel case. The numeric literal `252` is a
hit too. The Google section header `Returns:` and the Python keyword `return` are not hits.
"""

from __future__ import annotations

import ast
import io
import re
import tokenize
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src" / "hsmmkit"

# Everything that will become `hsmmkit` (Part II overview 5.3, rule 3).
SCOPE_PACKAGES = (
    "config",
    "data",
    "model",
    "statespace",
    "engine",
    "estimation",
    "inference",
    "evaluation",
    "persistence",
    "cli",
    "utils",
)
SCOPE_FILES = ("project.py", "__init__.py")

_TERMS = (
    r"prices?|returns|return_|log_?returns?|symbols?|tickers?|trading|providers?|strateg(?:y|ies)"
    r"|sharpe|drawdowns?|annualiz[a-z]*|bars_per_day"
)
_WORD = re.compile(rf"(?<![a-z0-9])({_TERMS})(?![a-z0-9])")
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_SECTION_HEADER = re.compile(r"^\s*Returns:\s*$", re.MULTILINE)
# The verb "returns" in prose ("Returns the mask", "it returns", "`f()` returns") is not a data name.
_VERB_RETURNS = re.compile(
    r"(?:^|[.;:!?]\s+|\n\s*|\b(?:it|this|that|which|then|also|function|method|call|never|only|always)\s+"
    r"|[`]\s*)returns\b(?!\s*(?:of|:|\())",
    re.IGNORECASE,
)
_LITERAL_252 = "252"


_CANON = {
    "prices": "price",
    "symbols": "symbol",
    "tickers": "ticker",
    "providers": "provider",
    "strategies": "strategy",
    "drawdowns": "drawdown",
    "logreturn": "log_return",
    "logreturns": "log_return",
    "log_returns": "log_return",
}


def term_of(match: str) -> str:
    """Canonical term of a raw match (`Prices` -> `price`, `annualized` -> `annualiz`)."""
    low = match.lower()
    return "annualiz" if low.startswith("annualiz") else _CANON.get(low, low)


def _terms_in_identifier(name: str) -> list[str]:
    return [term_of(m.group(1)) for m in _WORD.finditer(_CAMEL.sub("_", name).lower())]


def _terms_in_text(text: str) -> list[str]:
    text = _SECTION_HEADER.sub("", text)
    text = _VERB_RETURNS.sub(lambda m: m.group(0)[: -len("returns")] + "yields", text)
    lowered = text.lower()
    return [term_of(m.group(1)) for m in _WORD.finditer(lowered)]


def scan_source(source: str) -> list[tuple[int, str]]:
    """`(line, term)` hits of one module's source text, in order, without duplicates per line."""
    hits: list[tuple[int, str]] = []
    tree = ast.parse(source)

    def add(line: int, terms: list[str]) -> None:
        hits.extend((line, t) for t in terms)

    for node in ast.walk(tree):
        line = getattr(node, "lineno", 0)
        if isinstance(node, ast.Name):
            add(line, _terms_in_identifier(node.id))
        elif isinstance(node, ast.Attribute):
            add(line, _terms_in_identifier(node.attr))
        elif isinstance(node, ast.arg) or (isinstance(node, ast.keyword) and node.arg is not None):
            add(line, _terms_in_identifier(node.arg))
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            add(line, _terms_in_identifier(node.name))
        elif isinstance(node, ast.alias):
            add(line, _terms_in_identifier((node.asname or node.name).rsplit(".", 1)[-1]))
        elif isinstance(node, ast.Constant):
            if isinstance(node.value, str):
                add(line, _terms_in_text(node.value))
            elif (
                isinstance(node.value, int | float) and not isinstance(node.value, bool) and node.value == 252
            ):
                hits.append((line, _LITERAL_252))
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if tok.type == tokenize.COMMENT:
            add(tok.start[0], _terms_in_text(tok.string))
    # a term may be found by several routes (docstring node and token): keep one hit per line and term
    return sorted(set(hits))


def scope_files() -> list[Path]:
    """Every module in the neutral-name scope."""
    files: list[Path] = []
    for name in SCOPE_PACKAGES:
        files.extend(sorted((SRC / name).rglob("*.py")))
    files.extend(SRC / name for name in SCOPE_FILES if (SRC / name).is_file())
    return files


def scan_tree() -> list[str]:
    """All hits of the real tree as `path:line:term` (path relative to `src/hsmmkit`)."""
    out: list[str] = []
    for path in scope_files():
        rel = path.relative_to(SRC).as_posix()
        for line, term in scan_source(path.read_text(encoding="utf-8")):
            out.append(f"{rel}:{line}:{term}")
    return out


def entry_of(hit: str) -> str:
    """`path:line:term` -> allow-list key `path:term`."""
    path, _line, term = hit.rsplit(":", 2)
    return f"{path}:{term}"
