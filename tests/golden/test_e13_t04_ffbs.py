"""E13-T04: FFBS path frequencies against exact enumeration (AC1) and determinism / padding (AC3)."""

from __future__ import annotations

from typing import Any

import jax
import numpy as np
import pytest

from hsmmkit.data.batch import SequenceBatch
from hsmmkit.engine.expanded_hmm import _expanded_log_emis
from hsmmkit.engine.explicit_duration import _build_inputs
from hsmmkit.engine.ffbs import ffbs_expanded, ffbs_explicit
from hsmmkit.statespace.builders import initial_log_rho
from hsmmkit.utils.errors import ConfigError
from tests.helpers.engine_fixtures import make_explicit_scenario, make_scenario
from tests.reference import ref_ffbs

N_DRAWS = 200_000


def _frequencies(paths: np.ndarray, n_states: int) -> dict[tuple[int, ...], float]:
    """Empirical frequency of each path of `(n, T)` draws."""
    n, t_len = paths.shape
    code = np.zeros(n, dtype=np.int64)
    for t in range(t_len):
        code = code * n_states + paths[:, t]
    counts = np.bincount(code, minlength=n_states**t_len)
    out: dict[tuple[int, ...], float] = {}
    for c, k in enumerate(counts):
        digits = []
        for _ in range(t_len):
            digits.append(c % n_states)
            c //= n_states
        out[tuple(reversed(digits))] = k / n
    return out


def _check_within_3_se(freq: dict[tuple[int, ...], float], exact: dict[tuple[int, ...], float]) -> None:
    for path, p in exact.items():
        se = np.sqrt(max(p * (1 - p), 1e-12) / N_DRAWS)
        assert abs(freq[path] - p) <= 3 * se, (path, freq[path], p, 3 * se)


def test_e13_t04_expanded_paths_match_the_exact_posterior() -> None:
    """AC1: 200 000 draws of a 2-state, 4-step chain: all 16 path frequencies within 3 MC SE."""
    sc = make_scenario(2, [1, 1], 1, 4, seed=3)
    s, t_len = sc.space.n_expanded, 4
    trans = np.zeros((t_len, s, s))
    for t in range(1, t_len):
        trans[t] = np.exp(np.asarray(sc.space.transition_fn(sc.theta, None, None, sc.data.dt[t])[1]))
    ell = np.asarray(_expanded_log_emis(sc.theta, sc.spec, sc.space, sc.data))
    rho = np.exp(np.asarray(initial_log_rho(sc.spec, sc.space, sc.theta, sc.data.nominal_dt)))
    exact = ref_ffbs.expanded_path_probs(rho, trans, ell)
    pb = SequenceBatch.of(sc.data).padded()
    draws = np.asarray(ffbs_expanded(jax.random.key(0), sc.theta, sc.spec, sc.space, pb, N_DRAWS))
    assert draws.shape == (N_DRAWS, 1, 4)
    _check_within_3_se(_frequencies(draws[:, 0, :], s), exact)


def test_e13_t04_explicit_paths_match_the_exact_posterior() -> None:
    """AC1: the same for the explicit engine with geometric durations (truncated at D_max)."""
    sc = make_explicit_scenario(2, 4, 4, family="geometric", seed=5)
    inp = _build_inputs(sc.theta, sc.spec, sc.space, sc.data)
    rho = np.exp(np.asarray(inp.log_rho))
    pi = np.exp(np.asarray(inp.log_pi))
    pmf = np.exp(np.asarray(inp.log_pmf))
    ell = np.diff(np.asarray(inp.cum), axis=0)
    exact = ref_ffbs.explicit_path_probs(rho, pi, pmf, ell)
    pb = SequenceBatch.of(sc.data).padded()
    draws = np.asarray(ffbs_explicit(jax.random.key(1), sc.theta, sc.spec, sc.space, pb, N_DRAWS))
    assert draws.shape == (N_DRAWS, 1, 4)
    _check_within_3_se(_frequencies(draws[:, 0, :], 2), exact)


@pytest.mark.parametrize("engine", ["expanded", "explicit"])
def test_e13_t04_deterministic_per_key_with_padding_minus_one(engine: str) -> None:
    """AC3: same key -> same draws, another key differs; padded steps and padding sequences are `-1`."""
    if engine == "expanded":
        sc: Any = make_scenario(3, [2, 1, 2], 1, 60, n_cov_pi=2, n_cov_d=2, seed=2)
        fn = ffbs_expanded
    else:
        sc = make_explicit_scenario(3, 5, 60, family="negbin", seed=2)
        fn = ffbs_explicit
    seqs = (sc.data.window(0, 60), sc.data.window(10, 31), sc.data.window(40, 41))
    pb = SequenceBatch(seqs, (16, 64)).padded()
    a = np.asarray(fn(jax.random.key(7), sc.theta, sc.spec, sc.space, pb, 50))
    b = np.asarray(fn(jax.random.key(7), sc.theta, sc.spec, sc.space, pb, 50))
    c = np.asarray(fn(jax.random.key(8), sc.theta, sc.spec, sc.space, pb, 50))
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)
    assert a.shape == (50, pb.valid.shape[0], pb.valid.shape[1])
    valid = np.asarray(pb.valid)
    assert np.all(a[:, ~valid] == -1)
    assert np.all(a[:, valid] >= 0)
    assert np.all(a[:, 3:, :] == -1)  # the power-of-two padding sequence


def test_e13_t04_rejects_the_wrong_engine_and_bad_counts() -> None:
    """The engine tag and `n_samples` are validated."""
    sc = make_scenario(2, [1, 1], 1, 8, seed=1)
    pb = SequenceBatch.of(sc.data).padded()
    with pytest.raises(ConfigError, match="engine: explicit"):
        ffbs_explicit(jax.random.key(0), sc.theta, sc.spec, sc.space, pb, 3)
    with pytest.raises(ConfigError, match="n_samples"):
        ffbs_expanded(jax.random.key(0), sc.theta, sc.spec, sc.space, pb, 0)
