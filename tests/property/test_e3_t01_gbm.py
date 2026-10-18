"""E3-T01 AC4: sample moments within 3 SE; property checks of `log_prob`/`from_config`."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from regimekit.model.emission.gbm import GBMFamily, _broadcast_per_state, _prior_from_cfg
from regimekit.model.params import ParamManager
from regimekit.utils.errors import ConfigError

N_SIM = 20_000


def test_e3_t01_sample_moments_within_3se() -> None:
    """`sample` moments are within 3 SE of the analytic mean/sd for n=20000 per regime."""
    family = GBMFamily()
    mu, sigma = 0.10, 0.25
    dt_value = 1.0 / 252.0
    theta = {"emission.mu": jnp.array([[mu]]), "emission.sigma": jnp.array([[sigma]])}
    dt = jnp.full((N_SIM,), dt_value)
    regime = jnp.zeros((N_SIM,), dtype=jnp.int32)

    samples = np.asarray(family.sample(jax.random.key(0), regime, dt, theta, None))[:, 0]

    mean = (mu - 0.5 * sigma**2) * dt_value
    sd = sigma * np.sqrt(dt_value)
    se_mean = sd / np.sqrt(N_SIM)
    se_var = (sd**2) * np.sqrt(2.0 / N_SIM)

    assert abs(samples.mean() - mean) < 3 * se_mean
    assert abs(samples.var() - sd**2) < 3 * se_var


# --- property: log_prob shape/finiteness invariants across random draws -----------------------


@given(
    mu=st.floats(-0.5, 0.5),
    sigma=st.floats(0.01, 1.0),
    dt_value=st.floats(1.0 / 365.0, 5.0),
)
def test_e3_t01_log_prob_finite_and_matches_formula(mu: float, sigma: float, dt_value: float) -> None:
    """`log_prob` is finite and matches the closed-form `N((mu - sigma^2/2)*dt, sigma^2*dt)` density."""
    family = GBMFamily()
    theta = {"emission.mu": jnp.array([[mu]]), "emission.sigma": jnp.array([[sigma]])}
    dt = jnp.full((3,), dt_value)
    y = jnp.array([[0.0], [0.01], [-0.01]])

    out = np.asarray(family.log_prob(y, dt, theta, None))
    assert np.all(np.isfinite(out))

    mean = (mu - 0.5 * sigma**2) * dt_value
    sd = sigma * np.sqrt(dt_value)
    z = (np.asarray(y)[:, 0] - mean) / sd
    expected = -0.5 * z**2 - np.log(sd) - 0.5 * np.log(2.0 * np.pi)
    np.testing.assert_allclose(out[:, 0], expected, rtol=1e-9, atol=1e-9)


# --- from_config: configured family + specs are consumable by ParamManager ---------------------


def _emission_cfg_module():  # noqa: ANN202
    from regimekit.config import schema

    return schema


def test_e3_t01_from_config_builds_specs_param_manager_can_consume() -> None:
    """`GBMFamily.from_config` builds a family + specs; `ParamManager` round-trips `u`."""
    schema = _emission_cfg_module()
    cfg = schema.EmissionCfg(
        family="gbm",
        params={
            "mu": schema.ParamCfg(mode="fixed", values=[-0.3, 0.0, 0.2]),
            "sigma": schema.ParamCfg(mode="free", init=[0.12, 0.18, 0.35], constraint="increasing"),
        },
        covariates=schema.EmissionCovariates(drift=[], log_vol=["log_vix"]),
    )
    family, specs = GBMFamily.from_config(cfg, n_states=3, n_series=1, cov_names=("log_vix",))

    assert family.logvol_idx == (0,)
    names = {s.name for s in specs}
    assert names == {"emission.mu", "emission.sigma", "emission.delta"}

    manager = ParamManager(specs)
    u = manager.init_u()
    theta = manager.unflatten(u)
    np.testing.assert_allclose(np.asarray(theta["emission.mu"])[:, 0], [-0.3, 0.0, 0.2])
    assert theta["emission.sigma"].shape == (3,)
    assert theta["emission.delta"].shape == (3, 1)

    u_back = manager.flatten(theta)
    np.testing.assert_allclose(np.asarray(u_back), np.asarray(u), atol=1e-9)


def test_e3_t01_from_config_unknown_covariate_raises_config_error() -> None:
    """A drift/log-vol covariate name absent from `cov_names` raises `ConfigError`."""
    schema = _emission_cfg_module()
    cfg = schema.EmissionCfg(
        family="gbm",
        params={
            "mu": schema.ParamCfg(mode="free", init=[0.0]),
            "sigma": schema.ParamCfg(mode="free", init=[0.1]),
        },
        covariates=schema.EmissionCovariates(drift=["missing_cov"], log_vol=[]),
    )
    with pytest.raises(ConfigError) as exc_info:
        GBMFamily.from_config(cfg, n_states=1, n_series=1, cov_names=())
    assert exc_info.value.path == "model.emission.covariates.drift"


def test_e3_t01_from_config_drift_link_default_gamma_spec() -> None:
    """A drift covariate list with no `gamma` in `params` gets the default free/regularized spec."""
    schema = _emission_cfg_module()
    cfg = schema.EmissionCfg(
        family="gbm",
        params={
            "mu": schema.ParamCfg(mode="free", init=[0.0, 0.1]),
            "sigma": schema.ParamCfg(mode="free", init=[0.1, 0.2]),
        },
        covariates=schema.EmissionCovariates(drift=["term_spread"], log_vol=[]),
    )
    family, specs = GBMFamily.from_config(cfg, n_states=2, n_series=1, cov_names=("term_spread",))
    assert family.drift_idx == (0,)
    gamma = next(s for s in specs if s.name == "emission.gamma")
    assert gamma.shape == (2, 1)
    assert gamma.mode == "free"
    assert gamma.prior is not None


def test_e3_t01_from_config_explicit_gamma_and_delta_override() -> None:
    """An explicit `gamma`/`delta` entry in `params` overrides the default coefficient spec."""
    schema = _emission_cfg_module()
    cfg = schema.EmissionCfg(
        family="gbm",
        params={
            "mu": schema.ParamCfg(mode="free", init=[0.0]),
            "sigma": schema.ParamCfg(mode="free", init=[0.1]),
            "gamma": schema.ParamCfg(mode="fixed", values=[0.5]),
            "delta": schema.ParamCfg(mode="fixed", values=[0.25]),
        },
        covariates=schema.EmissionCovariates(drift=["a"], log_vol=["b"]),
    )
    family, specs = GBMFamily.from_config(cfg, n_states=1, n_series=1, cov_names=("a", "b"))
    assert family.drift_idx == (0,)
    assert family.logvol_idx == (1,)
    gamma = next(s for s in specs if s.name == "emission.gamma")
    delta = next(s for s in specs if s.name == "emission.delta")
    assert gamma.mode == "fixed"
    np.testing.assert_allclose(gamma.value, [[0.5]])
    assert delta.mode == "fixed"
    np.testing.assert_allclose(delta.value, [[0.25]])


def test_e3_t01_from_config_increasing_with_multiple_series_raises() -> None:
    """`constraint="increasing"` with `n_series > 1` raises rather than silently sharing sigma."""
    schema = _emission_cfg_module()
    cfg = schema.EmissionCfg(
        family="gbm",
        params={
            "mu": schema.ParamCfg(mode="free", init=[0.0, 0.1]),
            "sigma": schema.ParamCfg(mode="free", init=[0.1, 0.2], constraint="increasing"),
        },
        covariates=schema.EmissionCovariates(drift=[], log_vol=[]),
    )
    with pytest.raises(ConfigError):
        GBMFamily.from_config(cfg, n_states=2, n_series=2, cov_names=())


def test_e3_t01_from_config_values_shape_mismatch_raises() -> None:
    """A per-state `values`/`init` list with the wrong length raises `ConfigError`."""
    with pytest.raises(ConfigError):
        _broadcast_per_state([1.0, 2.0], n_states=3, n_series=1, flat=False)


# --- _prior_from_cfg: every PriorCfg type ------------------------------------------------------


def test_e3_t01_prior_from_cfg_normal_default_sd() -> None:
    """`type="normal"` with no `sd` defaults to `sd=1.0`."""
    schema = _emission_cfg_module()
    prior = _prior_from_cfg(schema.PriorCfg(type="normal", sd=None, alpha=None), None)
    assert prior.sd == 1.0  # type: ignore[attr-defined]


def test_e3_t01_prior_from_cfg_lognormal_uses_log_init_mean() -> None:
    """`type="lognormal"` centers `mean_log` at `log(init).mean()`."""
    schema = _emission_cfg_module()
    init = np.array([0.1, 0.2, 0.4])
    prior = _prior_from_cfg(schema.PriorCfg(type="lognormal", sd=0.5, alpha=None), init)
    assert prior.mean_log == pytest.approx(float(np.log(init).mean()))  # type: ignore[attr-defined]
    assert prior.sd == 0.5  # type: ignore[attr-defined]


def test_e3_t01_prior_from_cfg_dirichlet_default_alpha() -> None:
    """`type="dirichlet"` with no `alpha` defaults to `alpha=1.0`."""
    schema = _emission_cfg_module()
    prior = _prior_from_cfg(schema.PriorCfg(type="dirichlet", sd=None, alpha=None), None)
    assert prior.alpha == 1.0  # type: ignore[attr-defined]
