"""E9-T01 AC2: the `M = 1` loglik and smoothed posteriors are unchanged by the multi-series work.

The reference values in `m1_golden.json` were recorded on `main` before any E9-T01 change. They are
compared at 1e-12 (relative) so the test also holds on a different CPU/XLA build than the recording host;
the same-platform bitwise comparison is `test_e9_t01_m1_matches_the_legacy_emission_formula_bitwise`.
"""

from __future__ import annotations

import json

import jax.numpy as jnp
import numpy as np
import pytest

from regimekit.engine.expanded_hmm import _regime_log_emis
from tests.golden.e9_t01_m1_cases import CASES, GOLDEN_PATH, build, compute

GOLDEN = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", sorted(CASES))
def test_e9_t01_m1_loglik_and_posteriors_match_the_recorded_golden(name: str) -> None:
    """Recomputed loglik and smoothed posterior equal the values recorded on `main`."""
    ref = GOLDEN[name]
    got = compute(name)
    assert got["smoothed_shape"] == ref["smoothed_shape"]
    np.testing.assert_allclose(got["loglik"], float.fromhex(ref["loglik"]), rtol=1e-12, atol=0.0)
    expected = np.array([float.fromhex(h) for h in ref["smoothed_hex"]])
    actual = np.array([float.fromhex(h) for h in got["smoothed_hex"]])  # type: ignore[attr-defined]
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("name", sorted(CASES))
def test_e9_t01_m1_matches_the_legacy_emission_formula_bitwise(name: str) -> None:
    """For `M = 1` the engine's emission table equals the pre-E9-T01 formula bit for bit."""
    case = build(name)
    theta = case.manager.unflatten(case.u0)
    data = case.data
    y_safe = jnp.where(data.mask, data.observations, 0.0)
    z_e = jnp.zeros((y_safe.shape[0], 0)) if data.cov_emission is None else data.cov_emission
    legacy = jnp.where(
        jnp.all(data.mask, axis=-1)[:, None],
        case.spec.emission.log_prob(y_safe, data.dt, theta, z_e),
        0.0,
    )
    new = _regime_log_emis(theta, case.spec, data)
    assert np.asarray(new).tobytes() == np.asarray(legacy).tobytes()
