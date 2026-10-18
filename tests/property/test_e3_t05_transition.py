"""E3-T05 property checks: Delta->0 generator convergence, covariate link, from_config, jit/grad."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from regimekit.model.params import ParamManager
from regimekit.model.transition import JumpTransition, build_memory_layout, transition_matrix_generator
from regimekit.utils.errors import ConfigError


def _cfg_module():  # noqa: ANN202
    from regimekit.config import schema

    return schema


# --- AC4 property: generator vs first-order difference shrinks as dt -> 0 ----------------------


@given(dt_value=st.floats(1e-4, 1.0 / 12.0))
def test_e3_t05_generator_first_order_difference_bounded(dt_value: float) -> None:
    """`|expm(Q dt) - lambda*dt|` off-diagonal is bounded by `(lambda_max * dt)**2` for small `dt`."""
    theta = {"transition.pi_rows": jnp.array([[1.0], [1.0]])}
    lam = jnp.array([2.0, 6.0])
    dt = jnp.array([dt_value])
    p = np.asarray(transition_matrix_generator(theta, lam, dt))[0]

    lam_max_dt = 6.0 * dt_value
    assert abs(float(p[0, 1]) - 2.0 * dt_value) <= lam_max_dt**2 + 1e-12
    assert abs(float(p[1, 0]) - 6.0 * dt_value) <= lam_max_dt**2 + 1e-12


# --- covariate link -------------------------------------------------------------------------------


def test_e3_t05_zero_coefficients_reproduce_constant_pi() -> None:
    """`b = 0` reproduces the reference-only (uniform-logit) jump probabilities."""
    family = JumpTransition(cov_idx=(0,))
    layout = build_memory_layout(3, 1)
    r = layout.tuples.shape[0]
    theta = {
        "transition.a": jnp.zeros((r, 1)),
        "transition.b": jnp.zeros((r, 1, 1)),
    }
    z_pi = jnp.asarray(np.random.default_rng(0).normal(size=(5, 1)))
    out = np.asarray(family.jump_probs(theta, z_pi, layout))
    for t in range(1, 5):
        np.testing.assert_allclose(out[t], out[0], atol=1e-12)


def test_e3_t05_nonzero_coefficient_breaks_time_invariance() -> None:
    """Nonzero `b` with varying `z_pi` makes `jump_probs` vary over `t`."""
    family = JumpTransition(cov_idx=(0,))
    layout = build_memory_layout(3, 1)
    r = layout.tuples.shape[0]
    theta = {
        "transition.a": jnp.zeros((r, 1)),
        "transition.b": jnp.full((r, 1, 1), 2.0),
    }
    z_pi = jnp.array([[0.0], [1.0], [2.0]])
    out = np.asarray(family.jump_probs(theta, z_pi, layout))
    assert not np.allclose(out[0], out[1])
    assert not np.allclose(out[1], out[2])


# --- jump_probs_t sanity, jit/grad -----------------------------------------------------------------


def test_e3_t05_jump_probs_t_matches_jump_probs_single_step() -> None:
    """`jump_probs_t` at one covariate value matches `jump_probs`'s single-`t` slice."""
    family = JumpTransition(cov_idx=(0,))
    layout = build_memory_layout(3, 2)
    r = layout.tuples.shape[0]
    theta = {
        "transition.a": jnp.full((r, 1), 0.3),
        "transition.b": jnp.full((r, 1, 1), 0.5),
    }
    z_t = jnp.array([0.7])
    out_t = np.asarray(family.jump_probs_t(theta, z_t, layout))
    out_full = np.asarray(family.jump_probs(theta, z_t[None, :], layout))[0]
    np.testing.assert_allclose(out_t, out_full, atol=1e-12)


def test_e3_t05_jump_probs_t_no_covariates() -> None:
    """`jump_probs_t` with `transition.pi_rows` (no covariates) matches `jump_probs`'s slice."""
    family = JumpTransition()
    layout = build_memory_layout(3, 1)
    r = layout.tuples.shape[0]
    theta = {"transition.pi_rows": jnp.full((r, 2), 0.5)}
    out_t = np.asarray(family.jump_probs_t(theta, None, layout))
    out_full = np.asarray(family.jump_probs(theta, None, layout))[0]
    np.testing.assert_allclose(out_t, out_full, atol=1e-12)


def test_e3_t05_jump_probs_jittable_and_differentiable() -> None:
    """`jump_probs` is jittable and differentiable in `theta`."""
    family = JumpTransition()
    layout = build_memory_layout(3, 1)
    r = layout.tuples.shape[0]
    theta = {"transition.pi_rows": jnp.full((r, 2), 0.5)}

    jitted = jax.jit(lambda th: family.jump_probs(th, None, layout))
    out = jitted(theta)
    assert out.shape == (1, r, 3)

    grads = jax.grad(lambda th: jnp.sum(family.jump_probs(th, None, layout)))(theta)
    assert np.all(np.isfinite(np.asarray(grads["transition.pi_rows"])))


# --- from_config -----------------------------------------------------------------------------------


def test_e3_t05_from_config_no_covariates_simplex_spec() -> None:
    """`from_config` with no covariates builds a `transition.pi_rows` `simplex_ref` spec."""
    schema = _cfg_module()
    cfg = schema.TransitionCfg(mode="jump", order=1, covariates=[])
    family, specs = JumpTransition.from_config(cfg, n_states=3, order=1, cov_names=())
    assert family.cov_idx == ()
    (spec,) = specs
    assert spec.name == "transition.pi_rows"
    assert spec.domain == "simplex_ref"

    manager = ParamManager(specs)
    assert manager.n_free == 3 * 1


def test_e3_t05_from_config_with_dirichlet_prior() -> None:
    """`from_config` with no covariates and a Dirichlet prior sets `mode="prior"`."""
    schema = _cfg_module()
    cfg = schema.TransitionCfg(
        mode="jump", order=1, covariates=[], prior=schema.PriorCfg(type="dirichlet", sd=None, alpha=1.0)
    )
    _, (spec,) = JumpTransition.from_config(cfg, n_states=3, order=1, cov_names=())
    assert spec.mode == "prior"
    assert spec.prior is not None


def test_e3_t05_from_config_with_covariates_logit_specs() -> None:
    """`from_config` with covariates builds `transition.a`/`transition.b` logit specs."""
    schema = _cfg_module()
    cfg = schema.TransitionCfg(mode="jump", order=1, covariates=["log_vix", "term_spread"])
    family, specs = JumpTransition.from_config(cfg, n_states=3, order=1, cov_names=("log_vix", "term_spread"))
    assert family.cov_idx == (0, 1)
    names = {s.name for s in specs}
    assert names == {"transition.a", "transition.b"}
    b_spec = next(s for s in specs if s.name == "transition.b")
    assert b_spec.shape == (3, 1, 2)


def test_e3_t05_from_config_uses_coef_prior_sd() -> None:
    """`from_config` uses `cfg.coef_prior_sd`, not the hardcoded default, for `transition.b`'s prior."""
    schema = _cfg_module()
    cfg = schema.TransitionCfg(mode="jump", order=1, covariates=["log_vix"], coef_prior_sd=5.0)
    _, specs = JumpTransition.from_config(cfg, n_states=3, order=1, cov_names=("log_vix",))
    b_spec = next(s for s in specs if s.name == "transition.b")
    assert b_spec.prior is not None
    assert b_spec.prior.sd == 5.0  # type: ignore[attr-defined]


def test_e3_t05_from_config_dirichlet_with_covariates_sets_log_prior_extra() -> None:
    """A Dirichlet prior configured alongside covariates is not silently dropped."""
    schema = _cfg_module()
    cfg = schema.TransitionCfg(
        mode="jump",
        order=1,
        covariates=["log_vix"],
        prior=schema.PriorCfg(type="dirichlet", sd=None, alpha=2.0),
    )
    family, specs = JumpTransition.from_config(cfg, n_states=3, order=1, cov_names=("log_vix",))
    assert family.dirichlet_alpha == 2.0

    layout = build_memory_layout(3, 1)
    r = layout.tuples.shape[0]
    theta = {"transition.a": jnp.zeros((r, 1)), "transition.b": jnp.zeros((r, 1, 1))}
    log_prior = family.log_prior_extra(theta, layout)
    assert np.isfinite(float(log_prior))
    del specs


def test_e3_t05_log_prior_extra_is_zero_when_not_configured() -> None:
    """`log_prior_extra` is `0.` when no Dirichlet prior over covariate-logit `pi` is configured."""
    family = JumpTransition()
    layout = build_memory_layout(3, 1)
    assert float(family.log_prior_extra({}, layout)) == 0.0


def test_e3_t05_from_config_unknown_covariate_raises() -> None:
    """An unknown transition covariate name raises `ConfigError`."""
    schema = _cfg_module()
    cfg = schema.TransitionCfg(mode="jump", order=1, covariates=["missing"])
    with pytest.raises(ConfigError):
        JumpTransition.from_config(cfg, n_states=3, order=1, cov_names=())


def test_e3_t05_default_param_specs() -> None:
    """The config-free `param_specs` returns a single `transition.pi_rows` spec."""
    (spec,) = JumpTransition().param_specs(n_states=3, order=1, cov_names=("x",))
    assert spec.name == "transition.pi_rows"
    assert spec.shape == (3, 2)


def test_e3_t05_build_memory_layout_rejects_order_zero() -> None:
    """`build_memory_layout` rejects `order < 1`."""
    with pytest.raises(ConfigError):
        build_memory_layout(3, 0)
