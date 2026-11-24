"""E10-T01: the v0.2.0 golden values of the Appendix A pipeline, bitwise (CLAUDE.md section 3.9).

Never regenerate to make this pass; see `scripts/make_golden_v0_2_0.py` for the rules.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts.golden_env import PIN_FLAG, PINNED_ENV, pinned_environ
from scripts.make_golden_v0_2_0 import FILES, GOLDEN_DIR, ROOT, differences, load_stored

pytestmark = pytest.mark.finance


def test_e10_t01_golden_files_and_manifest_exist() -> None:
    """The five golden files and the manifest (versions, backend, pinned environment) exist."""
    for name in (*FILES, "manifest.json"):
        assert (GOLDEN_DIR / name).is_file(), name
    manifest = json.loads((GOLDEN_DIR / "manifest.json").read_text(encoding="utf-8"))
    for key in ("jax", "jaxlib", "numpy", "backend", "git_sha", "pinned_environment"):
        assert manifest[key]


def test_e10_t01_pipeline_reproduces_v0_2_0_bitwise() -> None:
    """AC1: the pipeline run with the instruction set pinned (a subprocess) equals the stored values."""
    out = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "make_golden_v0_2_0.py"), "--compare", str(GOLDEN_DIR)],
        cwd=ROOT,
        env=pinned_environ(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert out.stdout.strip().splitlines()[-1] == "DIFFERENCES: []", out.stdout[-2000:] + out.stderr[-2000:]


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


def test_e10_t01_pinned_environment_caps_the_instruction_set() -> None:
    """The pinned environment disables AVX-512 for numpy, glibc and XLA and sets the pin flag."""
    assert PINNED_ENV[PIN_FLAG] == "1"
    assert "AVX512F" in PINNED_ENV["NPY_DISABLE_CPU_FEATURES"]
    assert "-AVX512F" in PINNED_ENV["GLIBC_TUNABLES"]
    assert "--xla_cpu_max_isa=AVX2" in PINNED_ENV["XLA_FLAGS"]
    assert pinned_environ({"A": "1"})["A"] == "1"
