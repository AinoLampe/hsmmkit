"""E1-T05: model-side contracts -- frozen-ness, collapse helpers, pytree behaviour."""

from __future__ import annotations

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from regimekit.engine.base import InferenceEngine, Posterior
from regimekit.estimation.results import FitResult
from regimekit.evaluation.results import BacktestResult
from regimekit.inference.results import ForecastResult, PosteriorResult
from regimekit.model.duration.base import DurationFamily
from regimekit.model.emission.base import EmissionFamily
from regimekit.model.spec import ModelSpec
from regimekit.model.transition import MemoryLayout, TransitionModel
from regimekit.statespace.index import StateIndex


def _hand_built_index() -> StateIndex:
    """2 regimes, 3 phases each: S=6 expanded states, regimes {0,0,0,1,1,1}."""
    regime_of = np.array([0, 0, 0, 1, 1, 1])
    phase_of = np.array([1, 2, 3, 1, 2, 3])
    memory_of = np.zeros((6, 0), dtype=np.int64)
    collapse = np.zeros((6, 2))
    for s, r in enumerate(regime_of):
        collapse[s, r] = 1.0
    return StateIndex(
        regime_of=regime_of,
        phase_of=phase_of,
        memory_of=memory_of,
        collapse=collapse,
        row_of_state=regime_of.copy(),
        n_expanded=6,
        n_states=2,
        order=1,
    )


def _uniform_posterior(index: StateIndex, t: int = 4) -> Posterior:
    s = index.n_expanded
    return Posterior(
        log_alpha=jnp.log(jnp.ones((t, s)) / s),
        log_beta=jnp.zeros((t, s)),
        smoothed=jnp.ones((t, s)) / s,
        loglik=jnp.array(0.0),
        index=index,
    )


# --- AC2: regime_filtered()/regime_smoothed() on a hand-built 2-regime, 3-phase StateIndex ----


def test_e1_t05_regime_filtered_rows_sum_to_one() -> None:
    """`regime_filtered()` rows sum to 1."""
    post = _uniform_posterior(_hand_built_index())
    rows = jnp.sum(post.regime_filtered(), axis=-1)
    np.testing.assert_allclose(np.asarray(rows), np.ones(4), atol=1e-10)


def test_e1_t05_regime_smoothed_rows_sum_to_one() -> None:
    """`regime_smoothed()` rows sum to 1."""
    post = _uniform_posterior(_hand_built_index())
    rows = jnp.sum(post.regime_smoothed(), axis=-1)
    np.testing.assert_allclose(np.asarray(rows), np.ones(4), atol=1e-10)


def test_e1_t05_regime_filtered_matches_hand_computed_collapse() -> None:
    """`regime_filtered()` correctly collapses expanded-state mass onto its 2 regimes."""
    index = _hand_built_index()
    t = 1
    # Concentrate all mass on expanded state 3 (regime 1, phase 1).
    log_alpha = jnp.array([[-jnp.inf, -jnp.inf, -jnp.inf, 0.0, -jnp.inf, -jnp.inf]])
    post = Posterior(
        log_alpha=log_alpha,
        log_beta=jnp.zeros((t, 6)),
        smoothed=jax.nn.softmax(log_alpha, axis=-1),
        loglik=jnp.array(0.0),
        index=index,
    )
    filtered = post.regime_filtered()
    np.testing.assert_allclose(np.asarray(filtered), np.array([[0.0, 1.0]]), atol=1e-10)


def test_e1_t05_elapsed_posterior_shape_and_mass_conserved() -> None:
    """`elapsed_posterior()` has shape `(T, N, max_phase)` and conserves total mass per step."""
    post = _uniform_posterior(_hand_built_index())
    elapsed = post.elapsed_posterior()
    assert elapsed.shape == (4, 2, 3)
    np.testing.assert_allclose(np.asarray(jnp.sum(elapsed, axis=(1, 2))), np.ones(4), atol=1e-10)


# --- AC1: frozen dataclasses; Posterior/StateIndex usable as jit arguments/static closures ----


@pytest.mark.parametrize(
    "cls",
    [PosteriorResult, ForecastResult, FitResult, BacktestResult],
)
def test_e1_t05_result_dataclasses_are_frozen(cls: type[Any]) -> None:
    """`PosteriorResult`/`ForecastResult`/`FitResult`/`BacktestResult` are frozen dataclasses."""
    assert dataclasses.is_dataclass(cls)
    assert cls.__dataclass_params__.frozen  # type: ignore[attr-defined]


def test_e1_t05_posterior_is_frozen_dataclass() -> None:
    """`Posterior` is a frozen dataclass."""
    assert dataclasses.is_dataclass(Posterior)
    assert Posterior.__dataclass_params__.frozen


def test_e1_t05_state_index_field_values_are_immutable_arrays() -> None:
    """`StateIndex`'s array fields are JAX arrays; `eqx.Module` itself is frozen."""
    index = _hand_built_index()
    assert isinstance(index.regime_of, jax.Array)
    with pytest.raises(dataclasses.FrozenInstanceError):
        index.n_states = 99  # type: ignore[misc]


def test_e1_t05_state_index_usable_as_jit_argument() -> None:
    """`StateIndex` is usable as a `jax.jit` argument, with correct shape-based caching."""

    @jax.jit
    def total_states(index: StateIndex) -> jax.Array:
        return jnp.sum(index.collapse) + index.n_states

    index_a = _hand_built_index()
    assert float(total_states(index_a)) == pytest.approx(6.0 + 2.0)
    assert total_states._cache_size() == 1
    total_states(index_a)
    assert total_states._cache_size() == 1  # no recompilation for the identical index

    index_b = _hand_built_index()
    index_b = dataclasses.replace(index_b, regime_of=np.array([1, 1, 1, 0, 0, 0]))
    total_states(index_b)
    assert total_states._cache_size() == 1  # same shape/dtype -> reused, not retraced


def test_e1_t05_posterior_usable_as_jit_argument() -> None:
    """`Posterior` is usable as a `jax.jit` argument (leaves traced, `index` static aux)."""
    index = _hand_built_index()
    post = _uniform_posterior(index)

    @jax.jit
    def total_loglik(p: Posterior) -> jax.Array:
        return p.loglik + jnp.sum(p.smoothed)

    assert jnp.isfinite(total_loglik(post))
    assert total_loglik._cache_size() == 1

    post_same_index = _uniform_posterior(index)  # same `index` object -> _StaticRef.__eq__ True
    assert jnp.isfinite(total_loglik(post_same_index))
    assert total_loglik._cache_size() == 1  # reused: same index object, no recompilation

    other_index = _hand_built_index()  # a distinct index object -> _StaticRef.__eq__ False
    post_other_index = _uniform_posterior(other_index)
    assert jnp.isfinite(total_loglik(post_other_index))
    assert total_loglik._cache_size() == 2  # different index identity -> recompiled


def test_e1_t05_posterior_usable_as_static_closure() -> None:
    """`Posterior` can be closed over by a jitted function without being passed as an argument."""
    post = _uniform_posterior(_hand_built_index())

    @jax.jit
    def f(x: jax.Array) -> jax.Array:
        return x + post.loglik

    assert jnp.isfinite(f(jnp.array(1.0)))


def test_e1_t05_posterior_tree_flatten_unflatten_round_trips() -> None:
    """Flattening then unflattening a `Posterior` reconstructs it with the same `index` object."""
    index = _hand_built_index()
    post = _uniform_posterior(index)
    leaves, treedef = jax.tree_util.tree_flatten(post)
    assert len(leaves) == 4  # log_alpha, log_beta, smoothed, loglik
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert rebuilt.index is index
    np.testing.assert_allclose(np.asarray(rebuilt.smoothed), np.asarray(post.smoothed))


# --- ModelSpec: structure only, no data; Protocol stubs type-check (AC3 covered by `mypy`) -----


class _StubEmission:
    name = "stub"
    native_time = "continuous"
    support = "real"
    scale_by_dt = True

    def param_specs(self, n_states: object, n_series: object, cov_names: object) -> object:
        return ()

    def log_prob(self, y: object, dt: object, theta: object, z: object) -> object:
        return 0.0

    def sample(self, key: object, regime: object, dt: object, theta: object, z: object) -> object:
        return 0.0

    def init_carry(self) -> object:
        return None

    def step_carry(self, carry: object, y_t: object, theta: object) -> object:
        return carry

    def sufficient_stats(self, y: object, dt: object, gamma: object) -> object:
        return None


class _StubDuration:
    name = "stub"
    representation = "phase"
    supports_explicit = True

    def param_specs(self, n_states: object, cov_names: object) -> object:
        return ()

    def n_phases(self, i: object) -> object:
        return 1

    def advance_prob(self, theta: object, dt: object, z_d: object) -> object:
        return 0.0

    def advance_prob_t(self, theta: object, dt: float, z_d_t: object = None) -> object:
        return 0.0

    def log_pmf(self, theta: object, d_max: int, nominal_dt: float, z_entry: object) -> object:
        return 0.0

    def mean_sojourn(self, theta: object, nominal_dt: float) -> object:
        return 0.0

    def init_from_mean_sojourn(self, mean_steps: object, nominal_dt: float) -> object:
        return {}


class _StubTransition:
    mode = "jump"

    def param_specs(self, n_states: object, order: object, cov_names: object) -> object:
        return ()

    def jump_probs(self, theta: object, z_pi: object, memory: MemoryLayout) -> object:
        return 0.0

    def jump_probs_t(self, theta: object, z_pi_t: object, memory: MemoryLayout) -> object:
        return 0.0


def test_e1_t05_model_spec_holds_structure_only() -> None:
    """`ModelSpec` accepts protocol-conforming family stubs; it is a static structure only."""
    spec = ModelSpec(
        n_states=2,
        n_series=1,
        emission=_StubEmission(),
        duration=_StubDuration(),
        transition=_StubTransition(),
        order=1,
        duration_conditioning="current",
        engine="expanded",
        d_max=10,
        initial="uniform",
        param_specs=(),
    )
    assert jax.tree_util.tree_leaves(spec) == []  # everything is static: no dynamic leaves

    @jax.jit
    def f(x: jax.Array, s: ModelSpec) -> jax.Array:
        return x + s.n_states

    assert float(f(jnp.array(1.0), spec)) == pytest.approx(3.0)


def test_e1_t05_stub_families_satisfy_their_protocols() -> None:
    """The test stubs structurally satisfy `EmissionFamily`/`DurationFamily`/`TransitionModel`."""
    emission: EmissionFamily = _StubEmission()
    duration: DurationFamily = _StubDuration()
    transition: TransitionModel = _StubTransition()
    assert emission.support == "real"
    assert duration.supports_explicit is True
    assert transition.mode == "jump"


def test_e1_t05_inference_engine_protocol_is_a_protocol() -> None:
    """`InferenceEngine` is defined as a `Protocol` (signatures only, per the ticket)."""
    assert InferenceEngine.__mro__[1].__name__ == "Protocol" or hasattr(InferenceEngine, "_is_protocol")
