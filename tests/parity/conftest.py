"""Parity-suite plumbing: every parity test records its row; `HSMMKIT_WRITE_PARITY=1` writes the table."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import pytest

from tests.helpers.parity import ParityRow, render

_ROWS: list[ParityRow] = []
_TABLE = Path(__file__).resolve().parents[2] / "docs" / "generated" / "estimator_parity.md"


@pytest.fixture
def parity_log() -> Callable[[ParityRow], ParityRow]:
    """Record a `ParityRow` for the generated table (and hand it back for the assertion)."""

    def log(row: ParityRow) -> ParityRow:
        _ROWS.append(row)
        return row

    return log


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Write `docs/generated/estimator_parity.md` when asked (`scripts/gen_parity_table.py` does)."""
    del session
    if os.environ.get("HSMMKIT_WRITE_PARITY") == "1" and _ROWS and exitstatus in (0, 1):
        _TABLE.parent.mkdir(parents=True, exist_ok=True)
        _TABLE.write_text(render(_ROWS), encoding="utf-8")
