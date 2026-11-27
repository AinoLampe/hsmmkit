"""E13-T01a: the expanded E-step accumulators against the explicit-`xi` NumPy reference (AC1, AC2, AC4)."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

from hsmmkit.data.batch import SequenceBatch
from hsmmkit.engine.estep import expanded_estep
from hsmmkit.engine.expanded_hmm import ExpandedHMMEngine, _expanded_log_emis
from hsmmkit.statespace.builders import initial_log_rho
from hsmmkit.utils.errors import ConfigError
from tests.helpers.engine_fixtures import make_scenario
from tests.reference.ref_estep import estep_reference

ENGINE = ExpandedHMMEngine()


def _reference_sequence(sc: Any, data: Any) -> dict[str, Any]:  # noqa: ANN401
    """The reference accumulators of one dataset (inputs built with the library, reductions in NumPy)."""
    theta, space = sc.theta, sc.space
    t_len = int(data.observations.shape[0])
    ell = np.asarray(_expanded_log_emis(theta, sc.spec, space, data))
    log_rho = np.asarray(initial_log_rho(sc.spec, space, theta, data.nominal_dt))
    s = space.n_expanded
    log_a = np.zeros((t_len, s, s))
    z_pi = None if data.cov_transition is None else np.asarray(data.cov_transition)
    z_d = None if data.cov_duration is None else np.asarray(data.cov_duration)
    for t in range(1, t_len):
        log_a[t] = np.asarray(
            space.transition_fn(
                theta,
                None if z_pi is None else jnp.asarray(z_pi[t]),
                None if z_d is None else jnp.asarray(z_d[t]),
                data.dt[t],
            )[1]
        )
    idx = space.index
    return estep_reference(
        log_rho,
        log_a,
        ell,
        regime_of=np.asarray(idx.regime_of),
        rate_index_of=np.asarray(space.rate_index_of),
        is_last_phase=np.asarray(space.is_last_phase),
        exit_targets=np.asarray(space.exit_targets),
        row_of_state=np.asarray(idx.row_of_state),
        n_rates=space.n_rates,
        n_rows=int(space.entry_state_of.shape[0]),
        n_regimes=sc.spec.n_states,
    )


def _batch(sc: Any, cuts: list[tuple[int, int]]) -> tuple[SequenceBatch, list[Any]]:  # noqa: ANN401
    seqs = [sc.data.window(a, b) for a, b in cuts]
    return SequenceBatch(tuple(seqs), (16, 32, 64, 128)), seqs


CASES = [
    # (n_states, r, order, n_cov_pi, n_cov_d, seed)
    (2, [1, 2], 1, 0, 0, 1),
    (3, [2, 3, 1], 1, 2, 2, 2),
    (3, [1, 2, 2], 2, 0, 1, 3),
    (4, [2, 1, 3, 2], 1, 2, 0, 4),
    (3, [2, 2, 2], 2, 2, 2, 5),
]


@pytest.mark.parametrize(("n", "r", "order", "kp", "kd", "seed"), CASES)
def test_e13_t01a_accumulators_match_the_explicit_xi_reference(
    n: int, r: list[int], order: int, kp: int, kd: int, seed: int
) -> None:
    """AC1: every accumulator equals the sum of the reference ones (batch with padding), <= 1e-8."""
    sc = make_scenario(n, r, order, 150, n_cov_pi=kp, n_cov_d=kd, seed=seed)
    batch, seqs = _batch(sc, [(0, 60), (40, 90), (20, 33), (100, 101)])
    stats = expanded_estep(ENGINE, sc.theta, sc.spec, sc.space, batch.padded())
    refs = [_reference_sequence(sc, d) for d in seqs]
    tol = {"rtol": 0.0, "atol": 1e-8}
    for key in ("stay", "advance", "occupancy", "jump"):
        want = sum(np.asarray(ref[key]) for ref in refs)
        np.testing.assert_allclose(np.asarray(getattr(stats, key)), want, **tol)
    np.testing.assert_allclose(
        np.asarray(stats.gamma1)[0], sum(np.asarray(ref["gamma0"]) for ref in refs), **tol
    )
    for b, ref in enumerate(refs):
        n_b = int(seqs[b].observations.shape[0])
        assert abs(float(stats.loglik[b]) - ref["loglik"]) <= 1e-8 * abs(ref["loglik"])
        np.testing.assert_allclose(np.asarray(stats.gamma_regime[b, :n_b]), ref["gamma_regime"], **tol)
        assert np.all(np.asarray(stats.gamma_regime[b, n_b:]) == 0.0)
    assert np.all(np.asarray(stats.loglik[len(seqs) :]) == 0.0)  # the power-of-two padding sequences


@pytest.mark.parametrize(("n", "r", "order", "kp", "kd", "seed"), CASES)
def test_e13_t01a_invariants(n: int, r: list[int], order: int, kp: int, kd: int, seed: int) -> None:
    """AC2: regime marginals sum to 1, `stay + advance = occupancy`, jump mass = expected exits."""
    sc = make_scenario(n, r, order, 120, n_cov_pi=kp, n_cov_d=kd, seed=seed)
    batch, seqs = _batch(sc, [(0, 50), (30, 120), (10, 22)])
    stats = expanded_estep(ENGINE, sc.theta, sc.spec, sc.space, batch.padded())
    for b, d in enumerate(seqs):
        n_b = int(d.observations.shape[0])
        np.testing.assert_allclose(np.asarray(stats.gamma_regime[b, :n_b]).sum(axis=1), 1.0, atol=1e-10)
    np.testing.assert_allclose(
        np.asarray(stats.stay + stats.advance), np.asarray(stats.occupancy), rtol=0, atol=1e-10
    )
    # exits: the jump mass equals the advance mass of the last phases (no other state leaves its regime)
    last = np.asarray(sc.space.is_last_phase)
    rate = np.asarray(sc.space.rate_index_of)
    stay_adv_total = float(stats.advance.sum())
    assert float(stats.jump.sum()) <= stay_adv_total + 1e-10
    assert last.any()
    assert rate.max() < sc.space.n_rates
    refs = [_reference_sequence(sc, d) for d in seqs]
    np.testing.assert_allclose(
        float(stats.jump.sum()), sum(float(np.asarray(ref["jump"]).sum()) for ref in refs), atol=1e-8
    )


def test_e13_t01a_loglik_equals_the_engine_forward_pass() -> None:
    """The per-sequence log-likelihood is the engine's (1e-10 relative)."""
    sc = make_scenario(3, [2, 3, 2], 1, 90, n_cov_pi=2, n_cov_d=2, seed=8)
    stats = expanded_estep(ENGINE, sc.theta, sc.spec, sc.space, SequenceBatch.of(sc.data).padded())
    post = ENGINE.forward_backward(sc.theta, sc.spec, sc.space, sc.data)
    assert abs(float(stats.loglik[0]) - float(post.loglik)) <= 1e-10 * abs(float(post.loglik))
    np.testing.assert_allclose(
        np.asarray(stats.gamma_regime[0, :90]), np.asarray(post.regime_smoothed()), rtol=0, atol=1e-10
    )


def test_e13_t01a_tempering_one_is_the_untempered_pass_bitwise_and_tempering_changes_the_counts() -> None:
    """AC4: `beta = 1` through the tempered program equals the default bitwise; `beta < 1` flattens."""
    from hsmmkit.engine.estep import _estep_batch

    sc = make_scenario(3, [2, 2, 2], 1, 100, n_cov_pi=2, n_cov_d=2, seed=6)
    pb = SequenceBatch.of(sc.data).padded()
    base = expanded_estep(ENGINE, sc.theta, sc.spec, sc.space, pb)
    explicit = expanded_estep(ENGINE, sc.theta, sc.spec, sc.space, pb, beta_temper=1.0)
    through_tempered = _estep_batch(sc.theta, sc.spec, sc.space, pb, jnp.asarray(1.0), True)
    for key in ("loglik", "gamma_regime", "gamma1", "stay", "advance", "occupancy", "jump"):
        np.testing.assert_array_equal(np.asarray(getattr(base, key)), np.asarray(getattr(explicit, key)))
        np.testing.assert_array_equal(
            np.asarray(getattr(base, key)), np.asarray(getattr(through_tempered, key))
        )
    warm = expanded_estep(ENGINE, sc.theta, sc.spec, sc.space, pb, beta_temper=0.3)
    np.testing.assert_array_equal(np.asarray(warm.loglik), np.asarray(base.loglik))  # reported untempered
    assert float(np.abs(np.asarray(warm.stay) - np.asarray(base.stay)).max()) > 1e-6
    np.testing.assert_allclose(np.asarray(warm.stay + warm.advance), np.asarray(warm.occupancy), atol=1e-10)


def test_e13_t01a_rejects_bad_arguments() -> None:
    """`beta_temper` outside (0, 1] and the explicit engine are refused."""
    sc = make_scenario(2, [1, 1], 1, 30, seed=0)
    pb = SequenceBatch.of(sc.data).padded()
    for bad in (0.0, 1.5, -0.1):
        with pytest.raises(ConfigError, match="beta_temper"):
            expanded_estep(ENGINE, sc.theta, sc.spec, sc.space, pb, beta_temper=bad)


def test_e13_t01a_topology_and_labels_constrain_the_accumulators(tmp_path: Any) -> None:  # noqa: ANN401
    """AC1: a left-to-right topology (zero jump mass on the forbidden jumps) matches the reference."""
    from pathlib import Path

    from hsmmkit.model.params.manager import ParamManager
    from hsmmkit.statespace.builders import build_expanded
    from tests.golden.test_e12_t03_absorbing import _model

    _, data, spec = _model(Path(tmp_path), topology={"matrix": [[0, 1, 1], [0, 0, 1], [1, 0, 0]]})
    mgr = ParamManager(spec.param_specs)
    theta = mgr.unflatten(mgr.init_u())
    space = build_expanded(spec)

    class _Sc:  # the attributes `_reference_sequence` reads
        pass

    sc = _Sc()
    sc.theta, sc.spec, sc.space = theta, spec, space  # type: ignore[attr-defined]
    stats = expanded_estep(ENGINE, theta, spec, space, SequenceBatch.of(data).padded())
    ref = _reference_sequence(sc, data)
    for key in ("stay", "advance", "occupancy", "jump"):
        np.testing.assert_allclose(np.asarray(getattr(stats, key)), ref[key], rtol=0, atol=1e-8)
    jump = np.asarray(stats.jump)
    forbidden = np.array([[1, 0, 0], [1, 1, 0], [0, 1, 1]], dtype=bool)  # not allowed (incl. diagonal)
    assert np.all(jump[forbidden] < 1e-12)
    assert jump.sum() > 0


def test_e13_t01a_labels_restrict_the_regime_marginals_like_the_engine() -> None:
    """Supervision enters through the masked emissions: marginals and loglik equal the engine's."""
    import dataclasses

    sc = make_scenario(3, [2, 2, 2], 1, 80, n_cov_pi=2, n_cov_d=2, seed=12)
    labels = np.ones((80, 3), dtype=bool)
    labels[10:25, :] = False
    labels[10:25, 1] = True  # steps 10..24 must be regime 1
    labels[40:50, 0] = False
    data = dataclasses.replace(sc.data, labels=jnp.asarray(labels))
    stats = expanded_estep(ENGINE, sc.theta, sc.spec, sc.space, SequenceBatch.of(data).padded())
    post = ENGINE.forward_backward(sc.theta, sc.spec, sc.space, data)
    assert abs(float(stats.loglik[0]) - float(post.loglik)) <= 1e-10 * abs(float(post.loglik))
    got = np.asarray(stats.gamma_regime[0, :80])
    np.testing.assert_allclose(got, np.asarray(post.regime_smoothed()), rtol=0, atol=1e-10)
    assert np.all(got[10:25, 1] > 1 - 1e-10)
    assert np.all(got[40:50, 0] < 1e-12)
