"""E13-T04 AC2: regime frequencies of FFBS paths match the smoothed marginals within 3 MC SE at every step."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hsmmkit.data.batch import SequenceBatch
from hsmmkit.engine.expanded_hmm import ExpandedHMMEngine
from hsmmkit.engine.explicit_duration import ExplicitDurationEngine
from hsmmkit.engine.ffbs import ffbs_expanded, ffbs_explicit
from hsmmkit.model.params.manager import ParamManager
from hsmmkit.statespace.builders import build_expanded, build_explicit
from tests.golden.test_e12_t03_absorbing import _model
from tests.helpers.engine_fixtures import make_explicit_scenario, make_scenario

N_DRAWS = 30_000


def _labels(t_len: int, n: int, seed: int) -> jnp.ndarray:
    """Partial supervision: a few steps pinned to a regime, a few steps excluding one regime."""
    rng = np.random.default_rng(seed)
    labels = np.ones((t_len, n), dtype=bool)
    for t in rng.choice(t_len, size=4, replace=False):
        labels[t] = False
        labels[t, rng.integers(n)] = True
    for t in rng.choice(t_len, size=4, replace=False):
        if labels[t].sum() == n:
            labels[t, rng.integers(n)] = False
    return jnp.asarray(labels)


def _check(regimes: np.ndarray, smoothed: np.ndarray) -> None:
    """`regimes (n, T)` draws against `smoothed (T, N)`: every cell within 3 MC SE."""
    n_draw = regimes.shape[0]
    n = smoothed.shape[1]
    for k in range(n):
        freq = (regimes == k).mean(axis=0)
        p = smoothed[:, k]
        se = np.sqrt(np.maximum(p * (1 - p), 1e-12) / n_draw)
        bad = np.flatnonzero(np.abs(freq - p) > 3 * se + 1e-9)
        assert bad.size == 0, (k, bad, freq[bad], p[bad], 3 * se[bad])


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_e13_t04_expanded_frequencies_match_smoothed_marginals(seed: int) -> None:
    """Random expanded models (covariates, order 1/2, labels): path frequencies = smoothed marginals."""
    order = 1 + seed % 2
    sc = make_scenario(3, [2, 1, 2], order, 24, n_cov_pi=2, n_cov_d=1, seed=seed)
    data = dataclasses.replace(sc.data, labels=_labels(24, 3, seed))
    pb = SequenceBatch.of(data).padded()
    paths = np.asarray(ffbs_expanded(jax.random.key(seed), sc.theta, sc.spec, sc.space, pb, N_DRAWS))[:, 0, :]
    regimes = np.asarray(sc.space.index.regime_of)[paths]
    smoothed = np.asarray(
        ExpandedHMMEngine().forward_backward(sc.theta, sc.spec, sc.space, data).regime_smoothed()
    )
    _check(regimes, smoothed)
    assert np.all(
        regimes[:, _labels(24, 3, seed).sum(axis=1) == 1]
        == np.argmax(np.asarray(_labels(24, 3, seed)), axis=1)[_labels(24, 3, seed).sum(axis=1) == 1]
    )


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_e13_t04_explicit_frequencies_match_smoothed_marginals(seed: int) -> None:
    """Random explicit models (families, covariates, labels, left-censoring)."""
    family = ("negbin", "gamma", "geometric")[seed]
    # left-censoring builds its initial law at z = 0 (E12-T04), so that case has no jump covariates
    sc = make_explicit_scenario(3, 5, 24, family=family, n_cov_pi=0 if seed == 2 else 2, seed=seed)
    spec = dataclasses.replace(sc.spec, left_censored=bool(seed == 2))
    data = dataclasses.replace(sc.data, labels=_labels(24, 3, seed))
    pb = SequenceBatch.of(data).padded()
    paths = np.asarray(ffbs_explicit(jax.random.key(seed), sc.theta, spec, build_explicit(spec), pb, N_DRAWS))
    smoothed = np.asarray(
        ExplicitDurationEngine().forward_backward(sc.theta, spec, sc.space, data).regime_smoothed()
    )
    _check(paths[:, 0, :], smoothed)


def test_e13_t04_topology_is_respected(tmp_path: Path) -> None:
    """A left-to-right topology: no sampled path jumps backwards, and the frequencies match the marginals."""
    _, data, spec = _model(tmp_path, topology={"matrix": [[0, 1, 1], [0, 0, 1], [0, 0, 0]]})
    theta = ParamManager(spec.param_specs).unflatten(ParamManager(spec.param_specs).init_u())
    space = build_expanded(spec)
    pb = SequenceBatch.of(data).padded()
    paths = np.asarray(ffbs_expanded(jax.random.key(3), theta, spec, space, pb, N_DRAWS))[:, 0, :]
    regimes = np.asarray(space.index.regime_of)[paths]
    assert np.all(np.diff(regimes, axis=1) >= 0)
    smoothed = np.asarray(ExpandedHMMEngine().forward_backward(theta, spec, space, data).regime_smoothed())
    _check(regimes, smoothed)


@pytest.mark.parametrize("engine", ["expanded", "explicit"])
def test_e13_t04_padded_sequences_match_their_own_smoothed_marginals(engine: str) -> None:
    """Review: sequences of different lengths in one padded batch each follow their own posterior."""
    if engine == "expanded":
        sc: Any = make_scenario(3, [2, 1, 2], 1, 30, n_cov_pi=2, n_cov_d=1, seed=5)
        fn, eng = ffbs_expanded, ExpandedHMMEngine()
    else:
        sc = make_explicit_scenario(3, 5, 30, family="negbin", n_cov_pi=2, seed=5)
        fn, eng = ffbs_explicit, ExplicitDurationEngine()
    seqs = (sc.data.window(0, 30), sc.data.window(8, 21), sc.data.window(20, 24))
    pb = SequenceBatch(seqs, (32,)).padded()
    paths = np.asarray(fn(jax.random.key(11), sc.theta, sc.spec, sc.space, pb, N_DRAWS))
    for b, seq in enumerate(seqs):
        n_b = int(seq.observations.shape[0])
        draws = paths[:, b, :n_b]
        regimes = np.asarray(sc.space.index.regime_of)[draws] if engine == "expanded" else draws
        smoothed = np.asarray(eng.forward_backward(sc.theta, sc.spec, sc.space, seq).regime_smoothed())
        _check(regimes, smoothed)
        assert np.all(paths[:, b, n_b:] == -1)
