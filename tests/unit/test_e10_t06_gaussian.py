"""E10-T06: the core `gaussian` emission (diagonal, `scale_by_dt`, `ito_correction`, covariate links)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

from regimekit.config.schema import EmissionCfg, ParamCfg
from regimekit.model.build import EMISSION_FAMILIES
from regimekit.model.emission.gaussian import GaussianFamily
from regimekit.plugins import REGISTRY
from regimekit.utils.errors import ConfigError

N, T = 3, 50
MU = np.array([[0.05, -0.02], [0.01, 0.03], [-0.04, 0.0]])
SIGMA = np.array([[0.1, 0.2], [0.15, 0.25], [0.3, 0.4]])


def _theta(m: int) -> dict[str, jax.Array]:
    return {"emission.mean": jnp.asarray(MU[:, :m]), "emission.scale": jnp.asarray(SIGMA[:, :m])}


def _y(m: int, seed: int = 0) -> jax.Array:
    return jnp.asarray(np.random.default_rng(seed).normal(0.0, 0.2, size=(T, m)))


def test_e10_t06_registered_as_gaussian_in_the_core() -> None:
    """`gaussian` is a core family with a `from_config`; `gbm` is not part of the core registry entries."""
    entry = next(e for e in REGISTRY.entries("emission") if e.name == "gaussian")
    assert entry.provenance == "regimekit.core"
    assert EMISSION_FAMILIES["gaussian"] is GaussianFamily


@pytest.mark.parametrize("m", [1, 2])
def test_e10_t06_defaults_equal_scipy_norm_logpdf(m: int) -> None:
    """AC2: defaults (`scale_by_dt=False`, `ito_correction=False`) match `norm.logpdf` to 1e-12."""
    fam = GaussianFamily()
    y = _y(m)
    dt = jnp.full((T,), 0.37)  # ignored without scale_by_dt
    got = np.asarray(fam.log_prob_series(y, dt, _theta(m), None))  # (T, N, M)
    want = stats.norm.logpdf(np.asarray(y)[:, None, :], MU[None, :, :m], SIGMA[None, :, :m])
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        np.asarray(fam.log_prob(y, dt, _theta(m), None)), want.sum(axis=-1), rtol=0, atol=1e-12
    )


@pytest.mark.parametrize("dt_value", [1 / 252, 1 / 52, 1.0])
def test_e10_t06_delta_scaling_and_ito_term(dt_value: float) -> None:
    """With both options: mean `(mu - sigma^2/2) dt`, variance `sigma^2 dt` (checked against scipy)."""
    fam = GaussianFamily(scale_by_dt=True, ito_correction=True)
    y = _y(2)
    dt = jnp.full((T,), dt_value)
    got = np.asarray(fam.log_prob_series(y, dt, _theta(2), None))
    mean = (MU - 0.5 * SIGMA**2) * dt_value
    want = stats.norm.logpdf(np.asarray(y)[:, None, :], mean[None], (SIGMA * np.sqrt(dt_value))[None])
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-12)


def test_e10_t06_options_are_independent() -> None:
    """`scale_by_dt` alone has no Itô term; `ito_correction` alone has no dt scaling."""
    y, dt = _y(1), jnp.full((T,), 0.1)
    only_dt = GaussianFamily(scale_by_dt=True).log_prob_series(y, dt, _theta(1), None)
    want = stats.norm.logpdf(
        np.asarray(y)[:, None, :], MU[None, :, :1] * 0.1, SIGMA[None, :, :1] * np.sqrt(0.1)
    )
    np.testing.assert_allclose(np.asarray(only_dt), want, atol=1e-12)
    only_ito = GaussianFamily(ito_correction=True).log_prob_series(y, dt, _theta(1), None)
    want = stats.norm.logpdf(
        np.asarray(y)[:, None, :], (MU - 0.5 * SIGMA**2)[None, :, :1], SIGMA[None, :, :1]
    )
    np.testing.assert_allclose(np.asarray(only_ito), want, atol=1e-12)


def test_e10_t06_native_time_follows_scale_by_dt() -> None:
    """`native_time` is `continuous` with `scale_by_dt`, else `discrete`."""
    assert GaussianFamily().native_time == "discrete"
    assert GaussianFamily(scale_by_dt=True).native_time == "continuous"


def test_e10_t06_covariate_links_shift_mean_and_log_scale() -> None:
    """`mean = mu + gamma.z`, `log scale = log sigma + delta.z` over the columns of `mean_idx`/`scale_idx`."""
    fam = GaussianFamily(mean_idx=(0,), scale_idx=(1,))
    rng = np.random.default_rng(5)
    z = jnp.asarray(rng.normal(size=(T, 2)))
    gamma, delta = np.array([[0.3], [-0.2], [0.1]]), np.array([[0.2], [0.0], [-0.3]])
    theta = {**_theta(1), "emission.gamma": jnp.asarray(gamma), "emission.delta": jnp.asarray(delta)}
    y, dt = _y(1), jnp.ones(T)
    got = np.asarray(fam.log_prob_series(y, dt, theta, z))
    zn = np.asarray(z)
    mean = MU[None, :, :1] + zn[:, 0][:, None, None] * gamma[None, :, :1].transpose(0, 1, 2)
    scale = SIGMA[None, :, :1] * np.exp(zn[:, 1][:, None, None] * delta[None, :, :1])
    want = stats.norm.logpdf(np.asarray(y)[:, None, :], mean, scale)
    np.testing.assert_allclose(got, want, atol=1e-12)


@pytest.mark.parametrize("ito", [False, True])
def test_e10_t06_sample_moments_within_three_se(ito: bool) -> None:
    """Sampled values have the analytic mean and standard deviation (3 SE)."""
    fam = GaussianFamily(scale_by_dt=True, ito_correction=ito)
    n = 20000
    dt = jnp.full((n,), 0.5)
    regime = jnp.full((n,), 1, dtype=jnp.int32)
    x = np.asarray(fam.sample(jax.random.key(1), regime, dt, _theta(2), None))
    mean = (MU[1] - (0.5 * SIGMA[1] ** 2 if ito else 0.0)) * 0.5
    sd = SIGMA[1] * np.sqrt(0.5)
    assert np.all(np.abs(x.mean(axis=0) - mean) < 3 * sd / np.sqrt(n))
    assert np.all(np.abs(x.std(axis=0) / sd - 1) < 3 / np.sqrt(2 * n))


def _cfg(**kw: object) -> EmissionCfg:
    params = {"mean": ParamCfg(mode="free"), "scale": ParamCfg(mode="free")}
    return EmissionCfg(family="gaussian", params=params, **kw)  # type: ignore[arg-type]


def test_e10_t06_from_config_builds_options_and_param_names() -> None:
    """The config options reach the family; specs are named `emission.mean` / `emission.scale`."""
    fam, specs = GaussianFamily.from_config(_cfg(scale_by_dt=True, ito_correction=True), 3, 2, ())
    assert (fam.scale_by_dt, fam.ito_correction) == (True, True)
    assert [s.name for s in specs] == ["emission.mean", "emission.scale"]
    assert [s.shape for s in specs] == [(3, 2), (3, 2)]
    plain, _ = GaussianFamily.from_config(_cfg(), 3, 1, ())
    assert (plain.scale_by_dt, plain.ito_correction) == (False, False)


def test_e10_t06_from_config_covariate_keys_and_coefficient_specs() -> None:
    """`covariates: {mean, log_scale}` create `emission.gamma` / `emission.delta` with the right widths."""
    cfg = _cfg(covariates={"mean": ["a", "b"], "log_scale": ["b"]})
    fam, specs = GaussianFamily.from_config(cfg, 3, 1, ("a", "b"))
    assert (fam.mean_idx, fam.scale_idx) == ((0, 1), (1,))
    shapes = {s.name: s.shape for s in specs}
    assert shapes["emission.gamma"] == (3, 2)
    assert shapes["emission.delta"] == (3, 1)
    with pytest.raises(ConfigError) as info:
        GaussianFamily.from_config(cfg, 3, 1, ("a",))
    assert info.value.path == "model.emission.covariates.mean"


def test_e10_t06_from_config_requires_mean_and_scale() -> None:
    """A missing parameter config is a `ConfigError` naming the path."""
    cfg = EmissionCfg(family="gaussian", params={"mean": ParamCfg(mode="free")})
    with pytest.raises(ConfigError) as info:
        GaussianFamily.from_config(cfg, 3, 1, ())
    assert info.value.path == "model.emission.params.scale"


def test_e10_t06_increasing_scale_ordering_is_supported() -> None:
    """`constraint: increasing` on `scale` (M = 1) collapses it to an `(N,)` ordered spec."""
    cfg = EmissionCfg(
        family="gaussian",
        params={"mean": ParamCfg(mode="free"), "scale": ParamCfg(mode="free", constraint="increasing")},
    )
    _, specs = GaussianFamily.from_config(cfg, 3, 1, ())
    scale = next(s for s in specs if s.name == "emission.scale")
    assert scale.shape == (3,)
    assert scale.constraint == "increasing"


def test_e10_t06_sufficient_stats_use_the_family_time_scale() -> None:
    """Without `scale_by_dt` the weighted sums use `s = 1`, with it `s = dt`."""
    rng = np.random.default_rng(2)
    y = jnp.asarray(rng.normal(size=(30, 1)))
    dt = jnp.asarray(rng.uniform(0.5, 2.0, size=30))
    w = jnp.asarray(rng.dirichlet([1.0, 2.0], size=30))
    plain = GaussianFamily().sufficient_stats(y, dt, w)
    scaled = GaussianFamily(scale_by_dt=True).sufficient_stats(y, dt, w)
    np.testing.assert_allclose(np.asarray(plain["w_dt"]), np.asarray(plain["w"]))
    np.testing.assert_allclose(np.asarray(scaled["w_dt"]), np.asarray((w * dt[:, None]).sum(axis=0)))
    np.testing.assert_allclose(
        np.asarray(scaled["w_y2_dt"]), np.asarray((w * (y[:, :1] ** 2 / dt[:, None])).sum(axis=0))
    )
