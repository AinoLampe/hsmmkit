"""E10-T01: the v0.2.0 golden values of the Appendix A pipeline, bitwise (CLAUDE.md section 3.9).

Never regenerate to make this pass; see `scripts/make_golden_v0_2_0.py` for the rules.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.make_golden_v0_2_0 import FILES, GOLDEN_DIR, compute, differences, load_stored

pytestmark = pytest.mark.finance


@pytest.fixture(scope="module")
def fresh(tmp_path_factory: pytest.TempPathFactory) -> dict:  # type: ignore[type-arg]
    """Recompute the pipeline once per module."""
    return compute(tmp_path_factory.mktemp("golden"))


def test_e10_t01_golden_files_and_manifest_exist() -> None:
    """Golden files and manifest exist."""
    for name in (*FILES, "manifest.json"):
        assert (GOLDEN_DIR / name).is_file(), name
    manifest = json.loads((GOLDEN_DIR / "manifest.json").read_text(encoding="utf-8"))
    for key in ("jax", "jaxlib", "numpy", "backend", "git_sha"):
        assert manifest[key]


def test_e10_t01_pipeline_reproduces_v0_2_0_bitwise(fresh: dict) -> None:  # type: ignore[type-arg]
    """AC1: prepared data, fit, posterior, forecast samples and backtest metrics equal the stored values."""
    assert differences(load_stored(), fresh) == []


def test_e10_t01_one_flipped_bit_fails_the_comparison() -> None:
    """AC1: altering one bit of a stored array fails the comparison."""
    stored, altered = load_stored(), load_stored()
    arr = altered["fit.npz"]["uparams"].copy()
    arr.view(np.uint64)[0] ^= np.uint64(1)
    altered["fit.npz"]["uparams"] = arr
    assert differences(stored, stored) == []
    assert differences(stored, altered) == ["fit.npz:uparams"]


def test_e10_t01_manifest_matches_the_scripts_settings() -> None:
    """Manifest matches the scripts settings."""
    from scripts.make_golden_v0_2_0 import HORIZON, N_SIM, SEED

    manifest = json.loads((GOLDEN_DIR / "manifest.json").read_text(encoding="utf-8"))
    assert (manifest["horizon"], manifest["n_sim"], manifest["seed"]) == (HORIZON, N_SIM, SEED)
    assert isinstance(GOLDEN_DIR, Path)
