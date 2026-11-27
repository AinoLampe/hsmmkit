"""E13-T07 AC1: a deliberately broken M-step turns the corresponding parity row red."""

from __future__ import annotations

from pathlib import Path

import jax
import pytest

from hsmmkit.estimation import mstep
from tests.helpers.parity import evaluate_em_vs_map


def test_e13_t07_a_broken_m_step_fails_its_parity_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """EM with a duration M-step that is off by 20 % no longer reaches MAP's optimum: the row is `FAILED`."""
    good = evaluate_em_vs_map(tmp_path, "gaussian", "geometric", {})
    assert good.status == "ok"
    original = mstep.phase_mstep

    def broken(*args, **kwargs):  # type: ignore[no-untyped-def]  # noqa: ANN002, ANN003, ANN202
        out = dict(original(*args, **kwargs))
        out["duration.rate"] = 1.2 * out["duration.rate"]
        return out

    monkeypatch.setattr(mstep, "phase_mstep", broken)
    jax.clear_caches()  # the jitted EM step must be traced again with the broken M-step
    bad = evaluate_em_vs_map(tmp_path, "gaussian", "geometric", {})
    assert bad.status == "FAILED", bad
