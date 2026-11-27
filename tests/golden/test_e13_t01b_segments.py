"""E13-T01b: explicit-engine E-step (`eta`, `eta_censored`, `eta_first`, `xi`) against NumPy references."""

from __future__ import annotations

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.extend.core import jaxprs_in_params

from hsmmkit.data.batch import SequenceBatch
from hsmmkit.engine.estep_explicit import _estep_batch, explicit_estep
from hsmmkit.engine.explicit_duration import ExplicitDurationEngine, _build_inputs
from hsmmkit.utils.errors import ConfigError
from tests.helpers.engine_fixtures import make_explicit_scenario
from tests.reference import ref_hsmm_estep as ref

ENGINE = ExplicitDurationEngine()


def _inputs(sc: Any, data: Any) -> tuple[np.ndarray, ...]:  # noqa: ANN401
    """`(rho, pi, pmf, ell)` of one sequence in probability space, built by the library's input assembly."""
    inp = _build_inputs(sc.theta, sc.spec, sc.space, data)
    ell = np.diff(np.asarray(inp.cum), axis=0)
    return (
        np.exp(np.asarray(inp.log_rho)),
        np.exp(np.asarray(inp.log_pi)),
        np.exp(np.asarray(inp.log_pmf)),
        ell,
    )


def _stats(sc: Any, seqs: list[Any], edges: tuple[int, ...] = (16, 32, 64, 128)) -> Any:  # noqa: ANN401
    return explicit_estep(ENGINE, sc.theta, sc.spec, sc.space, SequenceBatch(tuple(seqs), edges).padded())


def _close(got: Any, want: Any, tol: float) -> None:  # noqa: ANN401
    np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=0, atol=tol)


@pytest.mark.parametrize(
    ("n", "d_max", "t_len", "family", "seed"),
    [(2, 3, 6, "negbin", 1), (3, 4, 8, "geometric", 2), (3, 3, 7, "gamma", 3), (2, 4, 8, "weibull", 4)],
)
def test_e13_t01b_matches_brute_force_enumeration(
    n: int, d_max: int, t_len: int, family: str, seed: int
) -> None:
    """AC1: `eta`, `eta_censored`, `xi` equal the sum over every segmentation (T <= 8), <= 1e-10."""
    sc = make_explicit_scenario(n, d_max, t_len, family=family, seed=seed)
    stats = _stats(sc, [sc.data])
    want = ref.brute_force(*_inputs(sc, sc.data))
    assert abs(float(stats.loglik[0]) - want["loglik"]) <= 1e-10
    for got, key in (
        (stats.eta, "eta"),
        (stats.eta_censored, "cens"),
        (stats.xi, "xi"),
        (stats.eta_first, "first"),
    ):
        _close(got, want[key], 1e-10)
    _close(stats.gamma_regime[0, :t_len], want["gamma"], 1e-10)
    _close(stats.rho_post[0], want["rho"], 1e-10)


def test_e13_t01b_left_censored_first_segment_goes_to_eta_first() -> None:
    """E12-T04: the closed first segment uses the residual pmf and is `eta_first`; brute force agrees."""
    sc = make_explicit_scenario(3, 4, 7, family="negbin", seed=5)
    spec = dataclasses.replace(sc.spec, left_censored=True)
    sc = dataclasses.replace(sc, spec=spec)
    stats = _stats(sc, [sc.data])
    want = ref.brute_force(*_inputs(sc, sc.data), first_residual=True)
    for got, key in (
        (stats.eta, "eta"),
        (stats.eta_censored, "cens"),
        (stats.eta_first, "first"),
        (stats.xi, "xi"),
    ):
        _close(got, want[key], 1e-10)
    assert float(np.asarray(stats.eta_first).sum()) > 0.0


@pytest.mark.parametrize(
    ("n", "d_max", "family", "kp", "seed"),
    [(3, 6, "negbin", 0, 1), (3, 8, "gamma", 2, 2), (4, 5, "geometric", 0, 3)],
)
def test_e13_t01b_matches_the_forward_backward_reference_on_random_batches(
    n: int, d_max: int, family: str, kp: int, seed: int
) -> None:
    """AC1: random instances up to T = 120 in a padded batch vs the NumPy forward-backward, <= 1e-8."""
    sc = make_explicit_scenario(n, d_max, 120, family=family, n_cov_pi=kp, seed=seed)
    cuts = [(0, 120), (10, 70), (30, 41), (50, 51)]
    seqs = [sc.data.window(a, b) for a, b in cuts]
    stats = _stats(sc, seqs)
    refs = [ref.forward_backward(*_inputs(sc, d)) for d in seqs]
    for got, key in ((stats.eta, "eta"), (stats.eta_censored, "cens"), (stats.xi, "xi")):
        _close(got, sum(np.asarray(r[key]) for r in refs), 1e-8)
    for b, r in enumerate(refs):
        n_b = int(seqs[b].observations.shape[0])
        assert abs(float(stats.loglik[b]) - r["loglik"]) <= 1e-8 * abs(r["loglik"])
        _close(stats.gamma_regime[b, :n_b], r["gamma"], 1e-8)
        assert np.all(np.asarray(stats.gamma_regime[b, n_b:]) == 0.0)
    _close(stats.rho_post[0], sum(np.asarray(r["rho"]) for r in refs), 1e-8)
    assert np.all(np.asarray(stats.loglik[len(seqs) :]) == 0.0)


def test_e13_t01b_mass_balance_and_jump_identity() -> None:
    """AC2: `sum d (eta + eta_censored + eta_first) = sum T_b`; `sum_j xi[i, j]` = completed of `i`."""
    sc = make_explicit_scenario(3, 6, 90, family="gamma", seed=7)
    spec = dataclasses.replace(sc.spec, left_censored=True)
    sc = dataclasses.replace(sc, spec=spec)
    seqs = [sc.data.window(0, 90), sc.data.window(20, 75), sc.data.window(40, 52)]
    stats = _stats(sc, seqs)
    d = np.arange(1, 7)[None, :]
    balance = float(
        (d * (np.asarray(stats.eta) + np.asarray(stats.eta_censored) + np.asarray(stats.eta_first))).sum()
    )
    assert balance == pytest.approx(90 + 55 + 12, abs=1e-8)
    completed = np.asarray(stats.eta).sum(axis=1) + np.asarray(stats.eta_first).sum(axis=1)
    _close(np.asarray(stats.xi).sum(axis=1), completed, 1e-8)


def test_e13_t01b_gamma_regime_equals_the_engine_smoothed_marginals() -> None:
    """AC3: the regime marginals are the E4-T06a ones, <= 1e-12."""
    sc = make_explicit_scenario(3, 5, 80, family="negbin", n_cov_pi=2, seed=9)
    stats = _stats(sc, [sc.data])
    post = ENGINE.forward_backward(sc.theta, sc.spec, sc.space, sc.data)
    _close(stats.gamma_regime[0, :80], post.regime_smoothed(), 1e-12)
    assert abs(float(stats.loglik[0]) - float(post.loglik)) <= 1e-12 * abs(float(post.loglik))


def test_e13_t01b_entry_covariates_give_per_start_counts() -> None:
    """With entry-time duration covariates `eta_entry` is per start and sums to `eta`; else `None`."""
    plain = make_explicit_scenario(3, 5, 60, family="negbin", seed=2)
    assert _stats(plain, [plain.data]).eta_entry is None
    sc = make_explicit_scenario(3, 5, 60, family="negbin", n_cov_d=2, seed=2)
    stats = _stats(sc, [sc.data])
    assert stats.eta_entry is not None
    assert stats.eta_entry_censored is not None
    entry = np.asarray(stats.eta_entry)
    assert entry.shape[-3:] == (64, 3, 5) or entry.shape[-3:] == (60, 3, 5)
    _close(entry.sum(axis=(0, 1)), stats.eta, 1e-9)
    _close(
        np.asarray(stats.eta_entry_censored).sum(axis=(0, 1)),
        np.asarray(stats.eta_censored).sum(axis=1),
        1e-9,
    )
    want = ref.forward_backward(*_inputs(sc, sc.data))
    _close(stats.eta, want["eta"], 1e-8)


def test_e13_t01b_no_t_n_d_array_without_entry_covariates() -> None:
    """AC4: the jaxpr (scan bodies included) holds no `(T, N, D_max)`-sized array."""
    sc = make_explicit_scenario(3, 5, 300, family="negbin", seed=1)
    pb = SequenceBatch.of(sc.data).padded()
    t_max, n, d = int(pb.observations.shape[1]), 3, 5
    jaxpr = jax.make_jaxpr(lambda th: _estep_batch.__wrapped__(th, sc.spec, sc.space, pb, False))(sc.theta)
    shapes: list[tuple[int, ...]] = []

    def walk(jp: Any) -> None:  # noqa: ANN401
        for eqn in jp.eqns:
            for var in [*eqn.invars, *eqn.outvars]:
                shapes.append(tuple(getattr(getattr(var, "aval", None), "shape", ())))
            for sub in jaxprs_in_params(eqn.params):
                walk(sub)

    walk(jaxpr.jaxpr)
    assert shapes
    assert not [sh for sh in shapes if int(np.prod(sh)) >= t_max * n * d and t_max in sh and d in sh]
    assert jnp.asarray(0).dtype is not None


def test_e13_t01b_rejects_the_expanded_engine_and_dd_jumps() -> None:
    """The expanded engine and duration-dependent jumps are refused with a `ConfigError`."""
    sc = make_explicit_scenario(3, 4, 20, seed=1)
    pb = SequenceBatch.of(sc.data).padded()
    with pytest.raises(ConfigError, match="engine: explicit"):
        explicit_estep(ENGINE, sc.theta, dataclasses.replace(sc.spec, engine="expanded"), sc.space, pb)
    with pytest.raises(ConfigError, match="duration-dependent"):
        explicit_estep(
            ENGINE, sc.theta, dataclasses.replace(sc.spec, duration_dependent_jumps=True), sc.space, pb
        )
