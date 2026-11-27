"""E13-T02b AC2: EM (closed form / ECM) reaches the MAP optimum from the same start (Appendix G)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from tests.helpers.parity import ParityRow, evaluate_em_vs_map

_ROWS: list[tuple[str, str, dict[str, Any]]] = [
    ("gaussian", "geometric", {}),
    ("gaussian", "negbin", {}),
    ("gaussian_full", "geometric", {}),
    ("student_t", "geometric", {}),
    ("poisson", "geometric", {}),
    ("poisson", "negbin", {}),
    ("categorical", "negbin", {}),
    ("gaussian", "geometric", {"topology": {"matrix": [[0, 1], [0, 0]]}}),
]


def _row_id(row: tuple[str, str, dict[str, Any]]) -> str:
    return f"{row[0]}-{row[1]}-{'ltr' if row[2] else 'full'}"


@pytest.mark.parametrize("row", _ROWS, ids=_row_id)
def test_e13_t02b_em_parity_with_map(
    tmp_path: Path, row: tuple[str, str, dict[str, Any]], parity_log: Callable[[ParityRow], ParityRow]
) -> None:
    """No priors, same start: parameters rel <= 1e-4, loglik rel <= 1e-6."""
    result = parity_log(evaluate_em_vs_map(tmp_path, *row))
    assert result.status == "ok", result
