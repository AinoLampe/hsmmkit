"""E13-T05b AC3: VB posterior means agree with MAP within 2 % for flat-ish priors (T >= 20 000)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from tests.helpers.parity import ParityRow, evaluate_vb_vs_map

_ROWS = [
    ("gaussian", "geometric"),
    ("gaussian_full", "geometric"),
    ("poisson", "negbin"),
    ("categorical", "geometric"),
]


@pytest.mark.parametrize(("family", "duration"), _ROWS, ids=[f"{f}-{d}" for f, d in _ROWS])
def test_e13_t05b_vb_posterior_means_match_map(
    tmp_path: Path, family: str, duration: str, parity_log: Callable[[ParityRow], ParityRow]
) -> None:
    """Every free parameter within 2 % (relative, floor 0.05 for entries near zero)."""
    result = parity_log(evaluate_vb_vs_map(tmp_path, family, duration))
    assert result.status == "ok", result
