"""E3-T03 property checks: `from_config`, hypothesis invariants, jit/grad, `advance_prob_t`."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from regimekit.finance.config import FinanceDurationCfg, FinanceParamCfg
from regimekit.model.duration.base import broadcast_per_state, prior_from_cfg
from regimekit.model.duration.geometric import GeometricDuration
from regimekit.model.duration.negbin import NegBinDuration
from regimekit.model.params import ParamManager
from regimekit.utils.errors import ConfigError

pytestmark = pytest.mark.finance  # needs the finance package (core variants: hsmmkit tickets)


def _cfg_module():  # noqa: ANN202
    from regimekit.config import schema

    return schema


# --- property: advance_prob is in (0, 1), log_pmf sums to ~1 -----------------------------------


@given(rate=st.floats(0.001, 20.0), dt_value=st.floats(1e-4, 5.0))
def test_e3_t03_advance_prob_in_unit_interval(rate: float, dt_value: float) -> None:
    """`advance_prob` is always in `(0, 1)` (excluding float64 saturation at `rate * dt >> 1`)."""
    family = GeometricDuration()
    theta = {"duration.rate": jnp.array([rate])}
    dt = jnp.full((1,), dt_value)
    a = float(family.advance_prob(theta, dt, None)[0, 0])
    assert 0.0 < a <= 1.0
    if rate * dt_value < 30.0:
        assert a < 1.0


@given(rate=st.floats(0.01, 5.0), r=st.integers(1, 5))
def test_e3_t03_log_pmf_sums_to_approximately_one(rate: float, r: int) -> None:
    """`log_pmf`'s exp sums to ~1 over `d = 1..D_max` (truncated + renormalized)."""
    family = NegBinDuration(r=(r,))
    theta = {"duration.rate": jnp.array([rate])}
    d_max = 200
    out = np.asarray(family.log_pmf(theta, d_max=d_max, nominal_dt=1.0 / 252.0, z_entry=None))
    total = np.exp(out[0]).sum()
    assert total == pytest.approx(1.0, abs=1e-6)


# --- jit / grad --------------------------------------------------------------------------------


def test_e3_t03_advance_prob_jittable_and_differentiable() -> None:
    """`advance_prob` is jittable and differentiable in `theta["duration.rate"]`."""
    family = NegBinDuration(r=(2, 3))
    theta = {"duration.rate": jnp.array([0.05, 0.1])}
    dt = jnp.full((5,), 1.0 / 252.0)

    jitted = jax.jit(lambda th: family.advance_prob(th, dt, None))
    out = jitted(theta)
    assert out.shape == (5, 2)

    grads = jax.grad(lambda th: jnp.sum(family.advance_prob(th, dt, None)))(theta)
    assert np.all(np.isfinite(np.asarray(grads["duration.rate"])))


def test_e3_t03_advance_prob_t_sanity() -> None:
    """`advance_prob_t` (E4-T02b's per-step wrapper) matches the no-covariate formula at one `dt`."""
    family = GeometricDuration()
    theta = {"duration.rate": jnp.array([0.05, 0.1])}
    out = np.asarray(family.advance_prob_t(theta, 1.0 / 252.0))
    expected = 1.0 - np.exp(-np.array([0.05, 0.1]) / 252.0)
    np.testing.assert_allclose(out, expected, atol=1e-12)


# --- from_config ---------------------------------------------------------------------------------


def test_e3_t03_from_config_geometric_builds_specs_param_manager_can_consume() -> None:
    """`GeometricDuration.from_config` builds specs consumable by `ParamManager`."""
    schema = _cfg_module()
    cfg = schema.DurationCfg(
        family="geometric",
        params={"rate": schema.ParamCfg(mode="free", init_mean_sojourn_steps=[20.0, 40.0])},
        covariates=["log_vix"],
    )
    family, specs = GeometricDuration.from_config(cfg, n_states=2, cov_names=("log_vix",), nominal_dt=1.0)
    assert family.cov_idx == (0,)
    names = {s.name for s in specs}
    assert names == {"duration.rate", "duration.c"}

    manager = ParamManager(specs)
    theta = manager.unflatten(manager.init_u())
    np.testing.assert_allclose(np.asarray(theta["duration.rate"]), [0.05, 0.025])


def test_e3_t03_from_config_negbin_requires_matching_r_length() -> None:
    """`negbin` with `r` of the wrong length raises `ConfigError`."""
    schema = _cfg_module()
    cfg = schema.DurationCfg(
        family="negbin",
        r=[3, 3],
        params={"rate": schema.ParamCfg(mode="free", init=[0.05, 0.05, 0.05])},
        covariates=[],
    )
    with pytest.raises(ConfigError):
        NegBinDuration.from_config(cfg, n_states=3, cov_names=(), nominal_dt=1.0)


def test_e3_t03_from_config_negbin_builds_specs() -> None:
    """`NegBinDuration.from_config` uses `cfg.r` and builds `duration.rate` from mean-sojourn init."""
    schema = _cfg_module()
    cfg = schema.DurationCfg(
        family="negbin",
        r=[3, 3, 2],
        params={"rate": schema.ParamCfg(mode="free", init_mean_sojourn_steps=[60.0, 120.0, 30.0])},
        covariates=[],
    )
    family, specs = NegBinDuration.from_config(cfg, n_states=3, cov_names=(), nominal_dt=1.0)
    assert family.r == (3, 3, 2)
    rate_spec = next(s for s in specs if s.name == "duration.rate")
    np.testing.assert_allclose(rate_spec.init, [3 / 60.0, 3 / 120.0, 2 / 30.0])


def test_e3_t03_from_config_init_mean_sojourn_days_rejected() -> None:
    """`init_mean_sojourn_days` is rejected here (needs `model/build.py`'s days->steps conversion)."""
    cfg = FinanceDurationCfg(
        family="geometric",
        params={"rate": FinanceParamCfg(mode="free", init_mean_sojourn_days=[60.0])},
        covariates=[],
    )
    with pytest.raises(ConfigError):
        GeometricDuration.from_config(cfg, n_states=1, cov_names=(), nominal_dt=1.0 / 252.0)


def test_e3_t03_from_config_unknown_covariate_raises() -> None:
    """A duration covariate name absent from `cov_names` raises `ConfigError`."""
    schema = _cfg_module()
    cfg = schema.DurationCfg(
        family="geometric",
        params={"rate": schema.ParamCfg(mode="free", init=[0.05])},
        covariates=["missing"],
    )
    with pytest.raises(ConfigError):
        GeometricDuration.from_config(cfg, n_states=1, cov_names=(), nominal_dt=1.0)


def test_e3_t03_from_config_plain_values_and_fixed_mode() -> None:
    """`from_config` also supports plain `values`/`init` (not just `init_mean_sojourn_steps`)."""
    schema = _cfg_module()
    cfg = schema.DurationCfg(
        family="geometric",
        params={"rate": schema.ParamCfg(mode="fixed", values=[0.05, 0.1])},
        covariates=[],
    )
    family, specs = GeometricDuration.from_config(cfg, n_states=2, cov_names=(), nominal_dt=1.0)
    del family
    rate_spec = specs[0]
    assert rate_spec.mode == "fixed"
    np.testing.assert_allclose(rate_spec.value, [0.05, 0.1])


def test_e3_t03_from_config_plain_values_shape_mismatch_raises() -> None:
    """A per-state `values` list with the wrong length raises `ConfigError`."""
    with pytest.raises(ConfigError):
        broadcast_per_state([1.0, 2.0], n_states=3)


def test_e3_t03_default_param_specs() -> None:
    """The config-free `param_specs` returns a single free/positive `duration.rate` spec."""
    (spec,) = GeometricDuration().param_specs(n_states=3, cov_names=("x",))
    assert spec.name == "duration.rate"
    assert spec.shape == (3,)
    assert spec.domain == "positive"
    assert spec.mode == "free"

    (spec_nb,) = NegBinDuration(r=(2, 2, 2)).param_specs(n_states=3, cov_names=())
    assert spec_nb.name == "duration.rate"


# --- prior_from_cfg: every PriorCfg type --------------------------------------------------------


def test_e3_t03_prior_from_cfg_normal_default_sd() -> None:
    """`type="normal"` with no `sd` defaults to `sd=1.0`."""
    schema = _cfg_module()
    prior = prior_from_cfg(schema.PriorCfg(type="normal", sd=None, alpha=None), None)
    assert prior.sd == 1.0  # type: ignore[attr-defined]


def test_e3_t03_prior_from_cfg_lognormal_and_dirichlet() -> None:
    """`type="lognormal"` centers on `log(init).mean()`; `type="dirichlet"` defaults `alpha=1.0`."""
    schema = _cfg_module()
    init = np.array([0.05, 0.1])
    log_prior = prior_from_cfg(schema.PriorCfg(type="lognormal", sd=0.5, alpha=None), init)
    assert log_prior.mean_log == pytest.approx(float(np.log(init).mean()))  # type: ignore[attr-defined]
    dir_prior = prior_from_cfg(schema.PriorCfg(type="dirichlet", sd=None, alpha=None), None)
    assert dir_prior.alpha == 1.0  # type: ignore[attr-defined]
