"""E13-T07: the GEM rows of the parity table (`gem_map`)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from tests.helpers.parity import ParityRow, evaluate_gem_vs_em, evaluate_gem_vs_map_gamma


def test_e13_t07_gem_geometric_matches_em_on_the_expanded_engine(
    tmp_path: Path, parity_log: Callable[[ParityRow], ParityRow]
) -> None:
    """Cross-engine parity (E13-T03a AC4)."""
    result = parity_log(evaluate_gem_vs_em(tmp_path))
    assert result.status == "ok", result


def test_e13_t07_gem_gamma_durations_match_map(
    tmp_path: Path, parity_log: Callable[[ParityRow], ParityRow]
) -> None:
    """Weighted-MLE duration updates reach the MAP optimum (E13-T03b AC2)."""
    result = parity_log(evaluate_gem_vs_map_gamma(tmp_path))
    assert result.status == "ok", result
