"""E9-T01: `ProductEmission` (independent per-series emissions, one regime process) and per-series masks."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import norm
from scipy.stats import t as student_t

from regimekit.data.contracts import PreparedDataset
from regimekit.engine.expanded_hmm import ExpandedHMMEngine, _regime_log_emis
from regimekit.model.build import build_model_spec
from regimekit.model.emission.gbm import GBMFamily
from regimekit.model.emission.product import ProductEmission
from regimekit.model.emission.student_t import StudentTFamily
from regimekit.model.params import ParamManager
from regimekit.statespace.builders import build_expanded
from tests.helpers.engine_fixtures import DT, make_dataset
from tests.helpers.two_series import SIGMA_STAR, two_series_config

T_LEN = 80


def _data(m: int = 2, seed: int = 0, mask: np.ndarray | None = None) -> PreparedDataset:
    rng = np.random.default_rng(seed)
    y = rng.normal(0.0, 0.01, size=(T_LEN, m))
    return make_dataset(y, np.full(T_LEN, DT), mask=mask)


def _gbm_theta() -> dict[str, jnp.ndarray]:
    return {
        "emission.mu": jnp.asarray([[0.08, 0.05], [-0.10, -0.05]]),
        "emission.sigma": jnp.asarray(SIGMA_STAR),
    }


def _scipy_log_prob(y: np.ndarray, theta: dict[str, jnp.ndarray]) -> np.ndarray:
    """`(T, N, M)` GBM log-density from scipy, one series at a time (independent of the family code)."""
    mu, sigma = np.asarray(theta["emission.mu"]), np.asarray(theta["emission.sigma"])
    mean = (mu - 0.5 * sigma**2) * DT  # (N, M)
    sd = sigma * np.sqrt(DT)
    return norm.logpdf(y[:, None, :], loc=mean[None], scale=sd[None])


# --- protocol delegation -------------------------------------------------------------------------


def test_e9_t01_product_delegates_the_family_protocol() -> None:
    """The wrapper reports the base family's identity and layout, and equals the base when unmasked."""
    base = GBMFamily()
    prod = ProductEmission.of(base, n_series=2)
    assert (prod.name, prod.native_time, prod.support, prod.scale_by_dt) == (
        base.name,
        base.native_time,
        base.support,
        base.scale_by_dt,
    )
    assert [s.shape for s in prod.param_specs(2, 2, ())] == [s.shape for s in base.param_specs(2, 2, ())]
    assert hash(prod) == hash(ProductEmission.of(GBMFamily(), n_series=2))

    y = jnp.asarray(_data().observations)
    theta = _gbm_theta()
    dt = jnp.full((T_LEN,), DT)
    assert np.asarray(prod.log_prob(y, dt, theta, None)).tobytes() == (
        np.asarray(base.log_prob(y, dt, theta, None)).tobytes()
    )
    assert prod.init_carry() is None


@pytest.mark.parametrize("family", ["gbm", "student_t"])
def test_e9_t01_log_prob_is_the_sum_of_per_series_densities_bitwise(family: str) -> None:
    """`log_prob == sum_m log_prob_series[..., m]` bit for bit, so `M = 1` results cannot move."""
    if family == "gbm":
        fam: GBMFamily | StudentTFamily = GBMFamily()
        theta = _gbm_theta()
    else:
        fam = StudentTFamily()
        theta = {
            "emission.m": jnp.asarray([[0.08, 0.05], [-0.10, -0.05]]),
            "emission.s": jnp.asarray(SIGMA_STAR),
            "emission.nu": jnp.asarray([5.0, 9.0]),
        }
    y = jnp.asarray(_data(seed=3).observations)
    dt = jnp.full((T_LEN,), DT)
    per_series = fam.log_prob_series(y, dt, theta, None)
    assert per_series.shape == (T_LEN, 2, 2)
    summed = np.asarray(jnp.sum(per_series, axis=-1))
    assert summed.tobytes() == np.asarray(fam.log_prob(y, dt, theta, None)).tobytes()


# --- AC3: per-series masks -----------------------------------------------------------------------


def test_e9_t01_masking_series_2_leaves_series_1_intact() -> None:
    """AC3: at a step where series 2 is masked the row equals series 1's own density (scipy, 1e-10)."""
    prod = ProductEmission.of(GBMFamily(), n_series=2)
    theta = _gbm_theta()
    data = _data(seed=5)
    y = np.asarray(data.observations)
    mask = np.ones((T_LEN, 2), dtype=bool)
    mask[10, 1] = False  # series 2 only
    mask[20, 0] = False  # series 1 only
    mask[30] = False  # both
    y_dirty = y.copy()
    y_dirty[~mask] = np.nan  # masked prints are never read

    dt = jnp.full((T_LEN,), DT)
    got = np.asarray(prod.log_prob_masked(jnp.where(mask, y_dirty, 0.0), jnp.asarray(mask), dt, theta, None))

    ref = _scipy_log_prob(y, theta)  # (T, N, M)
    expected = np.where(mask[:, None, :], ref, 0.0).sum(-1)
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-10)
    np.testing.assert_allclose(got[10], ref[10, :, 0], rtol=0, atol=1e-10)  # series 1 alone
    np.testing.assert_allclose(got[20], ref[20, :, 1], rtol=0, atol=1e-10)  # series 2 alone
    np.testing.assert_array_equal(got[30], 0.0)
    np.testing.assert_allclose(got[0], ref[0].sum(-1), rtol=0, atol=1e-10)  # untouched rows: full density


def test_e9_t01_student_t_masked_density_matches_scipy() -> None:
    """The per-series masking is family-agnostic: Student-t against scipy (1e-10)."""
    fam = StudentTFamily()
    prod = ProductEmission.of(fam, n_series=2)
    theta = {
        "emission.m": jnp.asarray([[0.08, 0.05], [-0.10, -0.05]]),
        "emission.s": jnp.asarray(SIGMA_STAR),
        "emission.nu": jnp.asarray([5.0, 9.0]),
    }
    y = np.asarray(_data(seed=7).observations)
    mask = np.ones((T_LEN, 2), dtype=bool)
    mask[4, 1] = False
    dt = jnp.full((T_LEN,), DT)
    got = np.asarray(prod.log_prob_masked(jnp.asarray(y), jnp.asarray(mask), dt, theta, None))
    loc = np.asarray(theta["emission.m"]) * DT
    scale = SIGMA_STAR * np.sqrt(DT)
    nu = np.asarray(theta["emission.nu"])[None, :, None]
    ref = student_t.logpdf(y[:, None, :], df=nu, loc=loc[None], scale=scale[None])
    expected = np.where(mask[:, None, :], ref, 0.0).sum(-1)
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-10)


def test_e9_t01_engine_emission_table_uses_per_series_masks() -> None:
    """`_regime_log_emis` keeps the observed series of a partly masked step (the old rule zeroed it)."""
    cfg = two_series_config()
    mask = np.ones((T_LEN, 2), dtype=bool)
    mask[10, 1] = False
    data = _data(seed=2, mask=mask)
    spec = build_model_spec(cfg, data)
    assert isinstance(spec.emission, ProductEmission)
    manager = ParamManager(spec.param_specs)
    theta = manager.unflatten(manager.init_u())
    table = np.asarray(_regime_log_emis(theta, spec, data))
    ref = _scipy_log_prob(np.asarray(data.observations), theta)
    np.testing.assert_allclose(table[10], ref[10, :, 0], rtol=0, atol=1e-10)
    np.testing.assert_allclose(table[11], ref[11].sum(-1), rtol=0, atol=1e-10)


def test_e9_t01_loglik_ignores_a_series_that_is_masked_everywhere() -> None:
    """`M = 2` with series 2 fully masked equals the `M = 1` model on series 1 (1e-10)."""
    cfg2, cfg1 = two_series_config(), two_series_config(symbols=("A",))
    data2 = _data(seed=9)
    data2 = dataclasses.replace(data2, mask=jnp.asarray(np.c_[np.ones(T_LEN, bool), np.zeros(T_LEN, bool)]))
    data1 = make_dataset(np.asarray(data2.observations)[:, :1], np.full(T_LEN, DT))
    spec2, spec1 = build_model_spec(cfg2, data2), build_model_spec(cfg1, data1)
    assert not isinstance(spec1.emission, ProductEmission)  # M = 1 keeps the bare family

    man2 = ParamManager(spec2.param_specs)
    theta2 = dict(man2.unflatten(man2.init_u()))
    theta2["emission.mu"] = jnp.asarray([[0.08, 7.0], [-0.10, -3.0]])
    theta2["emission.sigma"] = jnp.asarray(SIGMA_STAR)
    theta1 = dict(theta2)
    theta1["emission.mu"], theta1["emission.sigma"] = (
        theta2["emission.mu"][:, :1],
        theta2["emission.sigma"][:, :1],
    )

    eng = ExpandedHMMEngine()
    post2 = eng.forward_backward(theta2, spec2, build_expanded(spec2), data2)
    post1 = eng.forward_backward(theta1, spec1, build_expanded(spec1), data1)
    assert float(post2.loglik) == pytest.approx(float(post1.loglik), abs=1e-10)
    np.testing.assert_allclose(np.asarray(post2.smoothed), np.asarray(post1.smoothed), rtol=0, atol=1e-10)


def test_e9_t01_loglik_and_gradient_ignore_masked_values_even_when_nan() -> None:
    """Masked prints may be NaN: loglik is finite, independent of them, and the gradient is finite."""
    cfg = two_series_config()
    mask = np.ones((T_LEN, 2), dtype=bool)
    mask[[5, 6, 40], 1] = False
    mask[[12], 0] = False
    data = _data(seed=4, mask=mask)
    spec = build_model_spec(cfg, data)
    space, eng = build_expanded(spec), ExpandedHMMEngine()
    manager = ParamManager(spec.param_specs)
    u = manager.init_u()

    y = np.asarray(data.observations).copy()
    y_nan = y.copy()
    y_nan[~mask] = np.nan
    data_nan = dataclasses.replace(data, observations=jnp.asarray(y_nan))

    def nll(u_: jnp.ndarray, d: PreparedDataset) -> jnp.ndarray:
        return -eng.loglik(manager.unflatten(u_), spec, space, d)

    base = float(nll(u, data))
    assert float(nll(u, data_nan)) == pytest.approx(base, abs=1e-12)
    grad = jax.grad(nll)(u, data_nan)
    assert bool(jnp.all(jnp.isfinite(grad)))
    full = dataclasses.replace(data, mask=jnp.ones_like(data.mask))
    assert float(nll(u, full)) != pytest.approx(base, abs=1e-6)  # the mask does change the likelihood


# --- build: wrapping and per_series -----------------------------------------------------------------


def test_e9_t01_build_wraps_only_multi_series_models() -> None:
    """`n_series > 1` gets a `ProductEmission`; `n_series == 1` keeps the bare family (bitwise `M = 1`)."""
    spec2 = build_model_spec(two_series_config(), _data(2))
    spec1 = build_model_spec(two_series_config(symbols=("A",)), _data(1))
    assert isinstance(spec2.emission, ProductEmission)
    assert spec2.emission.n_series == 2
    assert isinstance(spec2.emission.base, GBMFamily)
    assert isinstance(spec1.emission, GBMFamily)
    shapes = {s.name: s.shape for s in spec2.param_specs}
    assert shapes["emission.mu"] == shapes["emission.sigma"] == (2, 2)


def test_e9_t01_per_series_false_shares_parameters_across_series() -> None:
    """`per_series: false` gives one `(N,)` value per regime, used for every series (ties across series)."""
    data = _data(2)
    spec = build_model_spec(two_series_config(per_series=False), data)
    shapes = {s.name: s.shape for s in spec.param_specs}
    assert shapes["emission.mu"] == shapes["emission.sigma"] == (2,)
    free_shared = ParamManager(spec.param_specs).n_free
    free_own = ParamManager(build_model_spec(two_series_config(), data).param_specs).n_free
    assert free_own - free_shared == 4  # mu and sigma lose one column of two regimes each

    prod = spec.emission
    theta = {"emission.mu": jnp.asarray([0.08, -0.10]), "emission.sigma": jnp.asarray([0.2, 0.4])}
    dt = jnp.full((T_LEN,), DT)
    y = jnp.asarray(data.observations)
    got = np.asarray(prod.log_prob_series(y, dt, theta, None))  # type: ignore[attr-defined]
    tiled = {k: jnp.stack([v, v], axis=1) for k, v in theta.items()}
    np.testing.assert_allclose(got, _scipy_log_prob(np.asarray(y), tiled), rtol=0, atol=1e-10)
    sample = prod.sample(jax.random.key(0), jnp.zeros(T_LEN, dtype=jnp.int32), dt, theta, None)
    assert sample.shape == (T_LEN, 2)  # one independent draw per series
    assert not np.allclose(np.asarray(sample[:, 0]), np.asarray(sample[:, 1]))


def test_e9_t01_increasing_sigma_needs_a_shared_parameter_for_several_series() -> None:
    """`constraint: increasing` is still refused for per-series sigma (M > 1) and allowed once shared."""
    from regimekit.utils.errors import ConfigError

    cfg = two_series_config(per_series=True)
    cfg.model.emission.params["sigma"] = cfg.model.emission.params["sigma"].model_copy(
        update={"constraint": "increasing"}
    )
    with pytest.raises(ConfigError):
        build_model_spec(cfg, _data(2))
    cfg_shared = two_series_config(per_series=False)
    cfg_shared.model.emission.params["sigma"] = cfg_shared.model.emission.params["sigma"].model_copy(
        update={"constraint": "increasing"}
    )
    spec = build_model_spec(cfg_shared, _data(2))
    assert {s.name: s.shape for s in spec.param_specs}["emission.sigma"] == (2,)
